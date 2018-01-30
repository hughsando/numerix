#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class GlobalPool : public Layer
{
   int srcW;
   int srcH;
   int channels;
   bool nchw;

public:
   GlobalPool( )
   {
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      if (sin0.size()!=3)
         TensorThrow("GlobalPool only supports H*W*C tensors");

      srcH = sin0[0];
      srcW = sin0[1];
      channels = sin0[2];

      Tensor *result = Tensor::makeBuffer(inBuffer, 1, 1, channels, inSrc0->type);

      nchw = inSrc0->isGpuNchw();

      float *dest = (float *)result->cpuWrite(nchw);
      const float *src = (const float *)inSrc0->cpuRead(nchw);

      std::vector<float> sum(channels);
      for(int y=0;y<srcH;y++)
      {
         const float *s = src + y*srcW*channels;
         for(int x=0;x<srcW;x++)
         {
            for(int c=0;c<channels;c++)
               sum[c] += *s++;
         }
      }

      float scale = 1.0/(srcW*srcH);
      for(int c=0;c<channels;c++)
         dest[c] = sum[c] * scale;

      return result;
   }

   /*
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
   */
};

Layer *Layer::createGlobalPool()
{
   return new GlobalPool();
}


} // end namespace numerix

