#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class Softmax : public Layer
{
   int srcW;
   int srcH;
   int channels;

public:
   Softmax( )
   {
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      if (sin0.size()!=3)
         TensorThrow("Softmax only supports H*W*C tensors");

      srcH = sin0[0];
      srcW = sin0[1];
      channels = sin0[2];

      startRun();

      Tensor *result = Tensor::makeBuffer(inBuffer, srcW, srcH, channels, inSrc0->type);

      float *dest = (float *)result->cpuWrite(false);
      const float *src = (const float *)inSrc0->cpuRead(false);
      std::vector<float> expChannel(channels);

      for(int y=0;y<srcH;y++)
      {
         const float *s = src + y*srcW*channels;
         float *d = dest + y*srcW*channels;
         for(int x=0;x<srcW;x++)
         {
            float sumExp = 0;
            for(int c=0;c<channels;c++)
            {
               float e = exp(*s++);
               expChannel[c] = e;
               sumExp += e;
            }
            float norm = 1.0/sumExp;
            for(int c=0;c<channels;c++)
               *d++ = expChannel[c]*norm;
         }
      }

      endRun();

      return result;
   }

   Tensor *src0;

   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      while(true)
      {
         int y = getNextJob();
         if (y>=srcH)
            break;
      }
   }
};

Layer *Layer::createSoftmax()
{
   return new Softmax();
}


} // end namespace numerix




