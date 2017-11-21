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
         TensorThrow("Pack - only supports H*W*C tensors");
      if (inSrc0->elementSize!=4)
         TensorThrow("Pack - only types of size4 supported");


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

      int dSdY = src0Stride[0];
      int sSdX = src0Stride[1];
      int dDdY = destStride[0];
      const int *srcP = (int *)src0->data;
      int *destP = (int *)destTensor->data;
      int srcChannels = src0->shape[2];

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         int srcY = y * stride;

         const int *sx = srcP + dSdY * srcY;
         int *dest = destP + dDdY*y;

         for(int x=0;x<destW;x++)
         {
            for(int ch=0;ch<srcChannels;ch++)
            {
                  for(int dx=0;dx<stride;dx++)
               for(int dy=0;dy<stride;dy++)
                  {
                     *dest++ = sx[ dx*sSdX + dy*dSdY ];
                  }
               sx++;
            }
            /*
            for(int dy=0;dy<stride;dy++)
            {
               memcpy(dest, sx + dy*dSdY, sSdX*stride);
               dest += sSdX*stride;
            }
            */
         }
         sx += sSdX*stride;
      }
   }
};

Layer *Layer::createPack(int inStride)
{
   return new Pack(inStride);
}


