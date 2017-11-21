#include <Tensor.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

using namespace numerix;

class Pack : public Layer
{
   int stride;

   int srcW;
   int srcH;
   int destW;
   int destH;
   int channels;

public:
   Pack(int inStride)
   {
      stride = inStride;
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      if (sin0.size()!=3)
         TensorThrow("Concat only supports H*W*C tensors");


      srcH = sin0[0];
      srcW = sin0[1];
      destW = srcW/stride;
      destH = srcH/stride;
      channels = sin0[2];

      Tensor *result = Tensor::makeBuffer(inBuffer, destW, destH, channels*stride*stride, inSrc0->type);

      src0 = inSrc0;
      destTensor = result;
      runThreaded();
      src0 = 0;
      destTensor = 0;

      return result;
   }

   Tensor *destTensor;
   Tensor *src0;

   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      int typeSize = src0->elementSize;
      const int *src0Stride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];
      int ds0x = src0Stride[1] * typeSize;

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         const u8 *s0 = src0->data + src0Stride[0] * (y*stride) * typeSize;
         u8 *d = destTensor->data + destStride[0] * y * typeSize;

         for(int x=0;x<destW;x++)
         {
            const u8 *s = s0;
            for(int dy=0;dy<stride;dy++)
            {
               memcpy(d, s, ds0x*stride);
               d += ds0x*stride;
               s+=  src0Stride[0]*typeSize;
            }
         }
      }
   }
};

Layer *Layer::createPack(int inStride)
{
   return new Pack(inStride);
}


