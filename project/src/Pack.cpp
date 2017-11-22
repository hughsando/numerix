#include <Tensor.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

using namespace numerix;

class Pack : public Layer
{
   int stride;

   std::vector<int> from;
   int srcW;
   int srcH;
   int srcChannels;
   int destW;
   int destH;
   int destChannels;

public:
   Pack(int inStride)
   {
      stride = inStride;
      srcW = srcH = srcChannels = -1;
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      if (sin0.size()!=3)
         TensorThrow("Pack - only supports H*W*C tensors");
      if (inSrc0->elementSize!=4)
         TensorThrow("Pack - only types of size4 supported");


      bool changed = srcH!=sin0[0] || srcW!=sin0[1] || srcChannels!=sin0[2];

      srcH = sin0[0];
      srcW = sin0[1];
      srcChannels = sin0[2];
      destW = srcW/stride;
      destH = srcH/stride;
      destChannels = srcChannels * stride * stride;

      Tensor *result = Tensor::makeBuffer(inBuffer, destW, destH, destChannels, inSrc0->type);

      if (changed)
         calcTransform();

      src0 = inSrc0;
      destTensor = result;
      runThreaded();
      src0 = 0;
      destTensor = 0;

      return result;
   }

   void reorg_cpu(int *x, int w, int h, int c, int stride, int *out)
   {
      bool forward = false;
      int b,i,j,k;
      int out_c = c/(stride*stride);

          for(k = 0; k < c; ++k){
              for(j = 0; j < h; ++j){
                  for(i = 0; i < w; ++i){
                      int in_index  = i + w*(j + h*k);
                      int c2 = k % out_c;
                      int offset = k / out_c;
                      int w2 = i*stride + offset % stride;
                      int h2 = j*stride + offset / stride;
                      int out_index = w2 + w*stride*(h2 + h*stride*c2);
                      // not forwards
                      out[in_index] = x[out_index];
                  }
              }
          }
   }

   void calcTransform()
   {
      int n = srcW * srcH * srcChannels;
      std::vector<int> srcIdx(n);

      // Fill srcIdx in darknet order (channel major) with numerix indices to pull from
      int idx = 0;
      for(int c=0;c<srcChannels;c++)
         for(int y=0;y<srcH;y++)
            for(int x=0;x<srcW;x++)
               // Write numerix src index...
               srcIdx[idx++] = y*srcW*srcChannels + x*srcChannels + c;

      // Apply magic reorder code, that seems to be not-quite convolutional, but must match weights
      std::vector<int> out(n);
      reorg_cpu(&srcIdx[0], srcW, srcH, srcChannels, stride, &out[0]);

      // Push the channel-major results to numerix order
      from.resize(n);
      idx = 0;
      for(int c=0;c<destChannels;c++)
         for(int y=0;y<destH;y++)
            for(int x=0;x<destW;x++)
            {
               // Write numerix src index...
               from[y*destW*destChannels + x*destChannels + c ]  = out[idx++];
            }
   }

   Tensor *destTensor;
   Tensor *src0;

   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      const int *srcP = (const int *)src0->data;
      int *destP = (int *)destTensor->data;
      int rowLen = destW*destChannels;

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         int offset = y*rowLen;
         const int *f = &from[offset];
         int *d = destP + offset;
         for(int i=0;i<rowLen;i++)
            d[i] = srcP[f[i]];
      }
   }
};

Layer *Layer::createPack(int inStride)
{
   return new Pack(inStride);
}


