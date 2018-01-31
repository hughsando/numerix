#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class MaxPool : public Layer
{
   int        filterY;
   int        filterX;
   int        strideX;
   int        strideY;

   int        srcW;
   int        srcH;
   int        channels;
   int        destW;
   int        destH;
   int        padOx;
   int        padOy;

   Padding    padding;

public:
   MaxPool(int inSizeX, int inSizeY,
           int inStepX, int inStepY,
           Padding inPadding )
   {
      filterX = inSizeX;
      filterY = inSizeY;
      strideX = inStepX;
      strideY = inStepY;
      padding = inPadding;
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("MaxPool only supports Float32 tensors");

      CShape sin = inSrc0->shape;
      if (sin.size()!=3)
         TensorThrow("MaxPool only supports H*W*C tensors");

      srcH = sin[0];
      srcW = sin[1];
      channels = sin[2];

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
         // This seems to match caffe...
         destW = (srcW)/strideX;
         destH = (srcH)/strideY;
      }
      //printf("MaxPool %dx%d + (%d,%d) %d,%d\n", destW, destW,  filterX, filterY, padOx,padOy);

      startRun();
      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==channels;
      }
      Tensor *result = inBuffer;
      if (!match)
      {
         Shape s(3);
         s[0] = destH;
         s[1] = destW;
         s[2] = channels;
         result = new Tensor( Float32, s );
      }

      src0 = inSrc0;
      destTensor = result;
      src0->cpuRead();
      destTensor->cpuWrite();

      runThreaded();
      src0 = 0;
      destTensor = 0;

      endRun();

      return result;
   }

   Tensor *src0;
   Tensor *destTensor;


   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      const int *srcStride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];
      const float *sIn = (const float *)src0->cpuRead();

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;


         float *dest = (float *)destTensor->cpuWrite() + destW*channels*y;
         int srcY = y*strideY;

         int dyMin = std::max(padOy-srcY,0);
         int dyMax = std::min(srcH+padOy-srcY,filterY);

         for(int x=0;x<destW;x++)
         {
            int srcX = x*strideX;

            int dxMin = std::max(padOx-srcX,0);
            int dxMax = std::min(srcW+padOx-srcX,filterX);

            const float *s = sIn + (srcY+dyMin-padOy)*srcStride[0] + (srcX+dxMin-padOx)*srcStride[1];
            for(int o=0;o<channels;o++)
            {
               const float *chan0 = s + o;
               float m = *chan0;
               for(int dy=dyMin;dy<dyMax;dy++)
               {
                  const float *chan = chan0;
                  chan0 += srcStride[0];
                  for(int dx=dxMin; dx<dxMax;dx++)
                  {
                     float val = *chan;
                     chan += channels;
                     if (val>m) m = val;
                     break;
                  }
                     break;
               }
               *dest++ = m;
            }
         }
      }
   }
};

Layer *Layer::createMaxPool(int sizeX, int sizeY,
                            int stepX, int stepY,
                            Padding padding )
{
   return new MaxPool(sizeX, sizeY, stepX, stepY, padding);
}

} // end namespace numerix
