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
      float *dest = (float *)destTensor->data;
      const int *srcStride = &inSrc0->strides[0];
      const int *destStride = &destTensor->strides[0];

      for(int y=0;y<destH;y++)
      {
         int srcY = srcStride[0]*y;

         int diMin = std::max(padOy-srcY,0);
         int diMax = std::min(srcH+padOy-srcY,filterY);
         int rows = diMax-diMin;

         const float *destY = dest + destStride[0]*y;

         for(int x=0;x<destW;x++)
         {
            // TODO
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
