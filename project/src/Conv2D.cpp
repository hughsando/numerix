#include <Tensor.h>
#include <stdexcept>

class Conv2D : public Layer
{
   int        strideX;
   int        strideY;
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
      return inBuffer;
   }
};

Layer *Layer::createConv2D(int strideY, int strideX,
                    Activation activation, Padding padding,
                    Tensor *weights, Tensor *bias)
{
   return new Conv2D(strideY, strideX, activation, padding, weights, bias);
}
