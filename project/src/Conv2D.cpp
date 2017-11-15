#include <Tensor.h>
#include <stdexcept>
#include <algorithm>

class Conv2D : public Layer
{
   int        strideX;
   int        strideY;
   int        filterY;
   int        filterX;
   int        inputs;
   int        outputs;

   Activation activation;
   Padding    padding;
   Tensor     *weights;
   Tensor     *bias;

public:
   Conv2D(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
   {
      strideY = inStrideY;
      strideX = inStrideX;
      activation = inActivation;
      padding = inPadding;

      // Output x Height x Width x Input
      Shape s = inWeights->shape;
      if (s.size()!=4)
         TensorThrow("Invalid Conv2D weigth shape");

      outputs = s[0];
      filterY = s[1];
      filterX = s[2];
      inputs =  s[3];

      if (inBias && inBias->shape.size()!=1)
         TensorThrow("Conv2D - bias should be one dimensional");

      if (inBias && inBias->shape[0]!=outputs)
         TensorThrow("Conv2D - bias does not match output size");


      weights = inWeights->incRef();
      bias = inBias ? inBias->incRef() : 0;
   }
   ~Conv2D()
   {
      weights->decRef();
      if (bias)
         bias->decRef();
   }

   float dot(const float *w, const float *s, int n)
   {
      float sum = 0;
      for(int i=0;i<n;i++)
         sum += w[i]*s[i];
      return sum;
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

      int srcH = sin[0];
      int srcW = sin[1];

      int destH = 0;
      int destW = 0;
      int padOx = 0;
      int padOy = 0;

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
      Tensor *destTensor = inBuffer;
      if (!match)
      {
         Shape s(3);
         s[0] = destH;
         s[1] = destW;
         s[2] = outputs;
         destTensor = new Tensor( Float32, s );
      }
      const float *src = (const float *)inSrc0->data;
      const float *b = bias ? (const float *)bias->data : 0;
      const int *srcStride = &inSrc0->strides[0];
      const int *destStride = &destTensor->strides[0];

      float *dest = (float *)destTensor->data;
      const float *w0 = (float *)weights->data;
      int featureSize = filterX*filterY*inputs;

      if (filterX==1 && filterY==1)
      {
         const float *src = (const float *)inSrc0->data;
         for(int y=0;y<destH;y++)
            for(int x=0;x<destW;x++)
            {
               const float *w = w0;
               for(int o=0;o<outputs;o++)
               {
                  float sum = dot(w, src, featureSize);
                  if (b)
                     sum+=b[o];
                  if (activation==actRelu && sum<0)
                     sum = 0;
                  else if (activation==actSigmoid)
                     sum = 1.0 / (1.0 + exp(-sum));
                  *dest++ = sum;
                  w+=outputs;
               }
               src+=inputs;
            }
      }
      else
      {
         int filterW = filterX*inputs;
         std::vector<float> srcBuf(filterW*filterY+1);
         float *srcPtr = &srcBuf[0];
         int filterRow = filterW*sizeof(float);

         const float *sIn = (float *)inSrc0->data;
         for(int y=0;y<destH;y++)
         {
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
               for(int o=0;o<outputs;o++)
               {
                  float sum = dot(w, srcPtr, featureSize);
                  if (b)
                     sum+=b[o];
                  if (activation==actRelu && sum<0)
                     sum = 0;
                  else if (activation==actSigmoid)
                     sum = 1.0 / (1.0 + exp(-sum));

                  *dest++ = sum;
                  w+=outputs;
               }
            }
         }
      }

      return destTensor;
   }
};

Layer *Layer::createConv2D(int strideY, int strideX,
                    Activation activation, Padding padding,
                    Tensor *weights, Tensor *bias)
{
   return new Conv2D(strideY, strideX, activation, padding, weights, bias);
}
