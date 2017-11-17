#include <Tensor.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>


using namespace numerix;

class Conv2D : public Layer
{
   int        strideX;
   int        strideY;
   int        filterY;
   int        filterX;
   int        inputs;
   int        outputs;
   int        diSize;

   int        srcW;
   int        srcH;
   int        destW;
   int        destH;
   int        padOx;
   int        padOy;

   Activation activation;
   Padding    padding;
   Tensor     *weights;
   Tensor     *pweights;
   Tensor     *bias;

   std::vector <float *> srcBuffers;
   std::vector <float *> diBuffers;

public:
   Conv2D(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inPWeights, Tensor *inBias)
   {
      strideY = inStrideY;
      strideX = inStrideX;
      activation = inActivation;
      padding = inPadding;
      diSize = 0;

      // Output x Height x Width x Input
      CShape s = inWeights->shape;
      if (!inPWeights && s.size()!=4)
         TensorThrow("Invalid Conv2D weigth shape");
      if (inPWeights && s.size()!=3)
         TensorThrow("Invalid SeparableConv2D depthwise shape");

      if (inPWeights)
      {
         CShape p = inPWeights->shape;
         if (p.size()!=2)
            TensorThrow("Invalid SeparableConv2D pointwise shape");
         if (s[0]!=p[1])
            TensorThrow("SeparableConv2D mismatched weights");
         diSize = p[1];

         outputs = p[0];
         inputs =  s[0];
         filterY = s[1];
         filterX = s[2];
         if (filterX==1 && filterY==1)
            TensorThrow("SeparableConv2D 1x1 not supported");
      }
      else
      {
         outputs = s[0];
         filterY = s[1];
         filterX = s[2];
         inputs =  s[3];
      }

      if (inBias && inBias->shape.size()!=1)
         TensorThrow("Conv2D - bias should be one dimensional");

      if (inBias && inBias->shape[0]!=outputs)
         TensorThrow("Conv2D - bias does not match output size");


      weights = inWeights->incRef();
      pweights = inPWeights ? inPWeights->incRef() : 0;
      bias = inBias ? inBias->incRef() : 0;

      int threads = GetWorkerCount();
      for(int i=0;i<threads;i++)
      {
         srcBuffers.push_back( (float *)Tensor::allocData(filterX*filterY*inputs*sizeof(float)) );
         if (diSize)
            diBuffers.push_back( (float *)Tensor::allocData(diSize*sizeof(float)) );
      }
   }
   ~Conv2D()
   {
      for(int i=0;i<srcBuffers.size();i++)
         Tensor::freeData( (unsigned char *)srcBuffers[i] );
      for(int i=0;i<diBuffers.size();i++)
         Tensor::freeData( (unsigned char *)diBuffers[i] );

      weights->decRef();
      if (pweights)
         pweights->decRef();
      if (bias)
         bias->decRef();
   }


   
   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("Conv2D only supports Float32 tensors");

      CShape &sin = inSrc0->shape;
      if (sin.size()!=3)
         TensorThrow("Conv2D only supports H*W*C tensors");

      if (sin[2]!=inputs)
      {
         printf("sin : %d %d %d\n", sin[0], sin[1], sin[2]);
         printf("weights : %d %dx%d %d\n", outputs, filterY, filterX, inputs );
         TensorThrow("Conv2D - weights do not match the number of input channels");
      }

      srcH = sin[0];
      srcW = sin[1];

      destH = 0;
      destW = 0;
      padOx = 0;
      padOy = 0;

      if (padding==padSame)
      {
         destW = (srcW+strideX-1)/strideX;
         int padX = (destW - 1)*strideX + filterX - srcW;
         padOx = padX>>1;

         destH = (srcH+strideY-1)/strideY;
         int padY = (destH - 1)*strideY + filterY - srcH;
         padOy = padY>>1;
      }
      else // padValid
      {
         destW = (srcW-filterX+1 + strideX-1)/strideX;
         destH = (srcH-filterY+1 + strideY-1)/strideY;
      }

      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape &s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==outputs;
      }
      Tensor *result = inBuffer;
      if (!match)
      {
         Shape s(3);
         s[0] = destH;
         s[1] = destW;
         s[2] = outputs;
         result = new Tensor( Float32, s );
      }

      src0 = inSrc0;
      destTensor = result;
      runThreaded();
      src0 = 0;
      destTensor = 0;

      return result;
   }

   Tensor *src0;
   Tensor *destTensor;


   void runThread(int threadId)
   {
      const float *b = bias ? (const float *)bias->data : 0;
      const int *srcStride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];
      float *di = pweights ? &diBuffers[threadId][0] : 0;

      const float *w0 = (float *)weights->data;
      int featureSize = filterX*filterY*inputs;
      int filters = filterX*filterY;

      if (filters==1)
      {
         while(true)
         {
            int y = getNextJob();
            if (y>=destH)
               break;

            const float *src = (const float *)src0->data + srcW*inputs*y;
            float *dest = (float *)destTensor->data + destW*outputs*y;
            for(int x=0;x<destW;x++)
            {
               const float *w = w0;
               for(int o=0;o<outputs;o++)
               {
                  float sum = dot(b?b[o] : 0.0f, w, src, featureSize, activation);
                  *dest++ = sum;
                  w+=featureSize;
               }
               src+=inputs;
            }
         }
      }
      else
      {
         int filterW = filterX*inputs;
         float *srcPtr = &srcBuffers[threadId][0];
         int filterRow = filterW*sizeof(float);

         const float *sIn = (float *)src0->data;

         while(true)
         {
            int y = getNextJob();
            if (y>=destH)
               break;


            float *dest = (float *)destTensor->data + destW*outputs*y;
            int srcY = y;

            int dyMin = std::max(padOy-srcY,0);
            int dyMax = std::min(srcH+padOy-srcY,filterY);

            if (dyMin>0)
               memset(srcPtr,0,filterRow*dyMin);
            if (dyMax<filterY)
               memset(srcPtr+dyMax*filterW,0,filterRow*(filterY-dyMax) );

            for(int x=0;x<destW;x++)
            {
               int srcX = x;

               int dxMin = std::max(padOx-srcX,0);
               int dxMax = std::min(srcW+padOx-srcX,filterX);

               const float *s = sIn + (srcY+dyMin-padOy)*srcStride[0] + (srcX+dxMin-padOx)*srcStride[1];
               for(int dy=dyMin;dy<dyMax;dy++)
               {
                  float *sp = srcPtr + filterW*dy;

                  if (dxMin>0)
                  {
                     memset(sp, 0, dxMin*inputs*sizeof(float));
                     sp +=dxMin*inputs;
                  }

                  memcpy(sp, s, (dxMax-dxMin)*inputs*sizeof(float));
                  s+= srcStride[0];

                  if (dxMax<filterX)
                     memset(sp + (dxMax-dxMin)*inputs, 0, (filterX-dxMax)*inputs*sizeof(float));
               }


               const float *w = w0;
               if (pweights)
               {
                  for(int d=0;d<diSize;d++)
                  {
                     int srcOff = d; // todo - depth_multiplier > 1
                     di[d] = dotSkip(w, srcPtr+srcOff, filters, inputs);
                     w+=filters;
                  }

                  const float *w = (const float *)pweights->data;
                  for(int o=0;o<outputs;o++)
                  {
                     float sum = dot(b?b[o]:0.0f, w, di, diSize, activation);
                     *dest++ = sum;
                     w+=diSize;
                  }
               }
               else
               {
                  for(int o=0;o<outputs;o++)
                  {
                     float sum = dot(b?b[o]:0.0f, w, srcPtr, featureSize, activation);
                     *dest++ = sum;
                     w+=featureSize;
                  }
               }
            }
         }
      }
   }
};

Layer *Layer::createConv2D(int strideY, int strideX,
                    Activation activation, Padding padding,
                    Tensor *weights, Tensor *pweights, Tensor *bias)
{
   return new Conv2D(strideY, strideX, activation, padding, weights, pweights, bias);
}
