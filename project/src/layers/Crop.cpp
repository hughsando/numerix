#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class Crop : public Layer
{
   int destW;
   int destH;
   int srcW;
   int srcH;
   int channels;
   int offsetX;
   int offsetY;
   bool nchw;

public:
   Crop(int inOffsetX, int inOffsetY)
   {
      offsetX = inOffsetX;
      offsetY = inOffsetY;
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      CShape sin1 = inSrc1->shape;
      if (sin0.size()!=3 || sin1.size()!=3)
         TensorThrow("Crop only supports H*W*C tensors");

      if (sin0[2]!=sin1[2])
      {
         char buf[1000];
         sprintf(buf, "Crop - mismatch channel sizes %dx%dx%d + %dx%dx%d",
                 sin0[0], sin0[1],sin0[2], sin1[0], sin1[1], sin1[2] );
         TensorThrow(buf);
      }

      srcH = sin0[0];
      srcW = sin0[1];
      destH = sin1[0];
      destW = sin1[1];
      channels = sin0[2];

      if (offsetX<0 || offsetY<0 || destW+offsetX>srcW || destH+offsetY>srcH)
      {
         char buf[1000];
         sprintf(buf, "Crop - bad offsets (%d,%d) %dx%dx%d + %dx%dx%d", offsetX, offsetY,
                 sin0[0], sin0[1],sin0[2], sin1[0], sin1[1], sin1[2] );
         TensorThrow(buf);
      }

      startRun();
      Tensor *result = Tensor::makeBuffer(inBuffer, destW, destH, channels, inSrc0->type);

      nchw = inSrc0->isGpuNchw();

      destTensor = result;

      // Prep in single-thread mode
      src0 = inSrc0;
      src0->cpuRead(nchw);
      destTensor->cpuWrite(nchw);

      runThreaded();

      src0 = 0;
      destTensor = 0;

      endRun();

      return result;
   }

   Tensor *destTensor;
   Tensor *src0;

   void runThread(int threadId)
   {
      if (nchw)
         runThreadNchw(threadId);
      else
         runThreadNhwc(threadId);
   }

   void runThreadNhwc(int threadId)
   {
      const int typeSize = src0->elementSize;
      const int srcStride = src0->strides[0]*typeSize;
      const int xOff = src0->strides[1]*typeSize * offsetX;
      const u8 *s0 =  src0->cpuRead() + offsetY*srcStride + xOff;
      u8 *d0 =  destTensor->cpuWrite();
      const int destStride = destTensor->strides[0] * typeSize;

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         memcpy(d0+y*destStride,  s0+y*srcStride, destStride );
      }
   }

   void runThreadNchw(int threadId)
   {
      int typeSize = src0->elementSize;
      /*
      const u8 *s0 =  src0->cpuRead() + offsetY*srcStride + xOff;
      const u8 *d0 =  destTensor->cpuRead() + xOff;
      */
      const int destStride = destTensor->strides[1] * typeSize;
      const int srcStride = src0->strides[1]*typeSize;

      while(true)
      {
         int chan = getNextJob();
         if (chan>=channels)
            break;


         const u8 *s0 =  src0->cpuRead() + src0->strides[0]*chan*typeSize + offsetY*srcStride + offsetX*typeSize;
         u8 *d0 =  destTensor->cpuWrite() + destTensor->strides[0]*chan*typeSize;
         for(int y=0;y<destH;y++)
         {
            memcpy(d0+y*destStride,  s0+y*srcStride, destStride );
         }
      }
   }


};

Layer *Layer::createCrop(int inXOffset, int inYOffset)
{
   return new Crop(inXOffset, inYOffset);
}


} // end namespace numerix

