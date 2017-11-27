#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class Concat : public Layer
{
   int srcW;
   int srcH;
   int c0;
   int c1;
   int channels;

public:
   Concat( )
   {
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer)
   {
      if (inSrc0->type != inSrc1->type)
         TensorThrow("Concat - input types must match");

      CShape sin0 = inSrc0->shape;
      CShape sin1 = inSrc1->shape;
      if (sin0.size()!=3 || sin1.size()!=3)
         TensorThrow("Concat only supports H*W*C tensors");

      if (sin0[0]!=sin1[0] || sin0[0]!=sin1[0])
         TensorThrow("Concat - mismatch image sizes");

      srcH = sin0[0];
      srcW = sin0[1];
      c0 = sin0[2];
      c1 = sin1[2];
      channels = c0 + c1;
      //printf("Concat -> %d %d %d\n", srcW, srcH, channels);

      Tensor *result = Tensor::makeBuffer(inBuffer, srcW, srcH, channels, inSrc0->type);

      src0 = inSrc0;
      src1 = inSrc1;
      destTensor = result;
      runThreaded();
      src0 = 0;
      destTensor = 0;

      return result;
   }

   Tensor *destTensor;
   Tensor *src0;
   Tensor *src1;

   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      int typeSize = src0->elementSize;
      const int *src0Stride = &src0->strides[0];
      const int *src1Stride = &src1->strides[0];
      const int *destStride = &destTensor->strides[0];
      int ds0x = src0Stride[1] * typeSize;
      int ds1x = src1Stride[1] * typeSize;
      int ddx = destStride[1] * typeSize;

      while(true)
      {
         int y = getNextJob();
         if (y>=srcH)
            break;

         const u8 *s0 = src0->data + src0Stride[0] * y * typeSize;
         const u8 *s1 = src1->data + src1Stride[0] * y * typeSize;
         u8 *d = destTensor->data + destStride[0] * y * typeSize;

         for(int x=0;x<srcW;x++)
         {
            memcpy(d, s0, ds0x);
            s0 += ds0x;
            d += ds0x;
            memcpy(d, s1, ds1x);
            s1 += ds1x;
            d += ds1x;
         }
      }
   }
};

Layer *Layer::createConcat()
{
   return new Concat();
}


} // end namespace numerix
