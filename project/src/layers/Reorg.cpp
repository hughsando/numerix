#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

class Reorg : public Layer
{
   int stride;

   std::vector<int> fromNhwc;
   std::vector<int> fromNchw;

   int srcW;
   int srcH;
   int srcChannels;
   int destW;
   int destH;
   int destChannels;

public:
   Reorg(int inStride)
   {
      stride = inStride;
      srcW = srcH = srcChannels = -1;
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      CShape sin0 = inSrc0->shape;
      if (sin0.size()!=3)
         TensorThrow("Reorg - only supports H*W*C tensors");
      if (inSrc0->elementSize!=4)
         TensorThrow("Reorg - only types of size4 supported");


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

      bool nchw = src0->isGpuNchw();
      if (nchw && false)
      {
         reorg_cpu((int *)src0->cpuRead(nchw),  srcW, srcH, srcChannels, stride, (int *)destTensor->cpuWrite(nchw) );
      }
      else
      {
         // Perform conversion while single-threaded
         src0->cpuRead(nchw);
         destTensor->cpuWrite(nchw);
         runThreaded(true);
         src0 = 0;
         destTensor = 0;
      }

      return result;
   }

   void reorg_cpu(const int *x, int smallW, int smallH, int smallChannels, int stride, int *out)
   {
      bool forward = false;

      int bigW = smallW * stride;
      int bigH = smallH * stride;
      int bigChannels = smallChannels/(stride*stride);

          for(int ch = 0; ch < smallChannels; ++ch){
              for(int sy = 0; sy < smallH; ++sy){
                  for(int sx = 0; sx < smallW; ++sx){
                      int in_index  = sx + smallW*(sy + smallH*ch);

                      // Pull consecutive channels ...
                      int bigChannel = ch % bigChannels;
                      int offset = ch / bigChannels;
                      // from a pattern like:  0 1
                      //                       2 3
                      int bigX = sx*stride + offset % stride;
                      int bigY = sy*stride + offset / stride;

                      int out_index = bigX + bigW*(bigY + bigH*bigChannel);

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
      fromNhwc.resize(n);
      idx = 0;
      for(int c=0;c<destChannels;c++)
         for(int y=0;y<destH;y++)
            for(int x=0;x<destW;x++)
            {
               // Write numerix src index...
               fromNhwc[y*destW*destChannels + x*destChannels + c ]  = out[idx++];
            }

      #ifdef NX_GPU
      for(int i=0;i<n;i++)
         srcIdx[i] = i;
      fromNchw.resize(n);
      reorg_cpu(&srcIdx[0], srcW, srcH, srcChannels, stride, &fromNchw[0]);
      #endif
   }

   Tensor *destTensor;
   Tensor *src0;

   void runThread(int threadId)
   {
      runThreadMulti(threadId);
   }

   void runThreadMulti(int threadId)
   {
      bool nchw = src0->isGpuNchw();

      const int *srcP = (const int *)src0->cpuRead(nchw);
      int srcRow = src0->strides[0];
      int *destP = (int *)destTensor->cpuWrite(nchw);
      int rowLen = destW*destChannels;

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         int offset = y*rowLen;
         int *d = destP + offset;

         if (false && !nchw)
         {
            // - this is not valid because the srcW/srcH is passed to reorg, not destW/destH
            //
            // The dest image size is smaller than src size, but has more channels
            //  It concats 4 'stacks' of channels into 1 channel, pulled according to the high-res XY:
            //  0 1
            //  2 3
            //
            const int *src0 = srcP + srcRow*y*stride;
            for(int x=0;x<destW;x++)
            {
               for(int dy=0;dy<stride;dy++)
               {
                  const int *sy = src0 + srcRow*dy;
                  memcpy(d, sy, stride*srcChannels*sizeof(int));
                  d+= srcChannels*stride;
               }
               src0 += stride*srcChannels;
            }
         }
         else
         {
            const int *f = nchw ? &fromNchw[offset] : &fromNhwc[offset];
            for(int i=0;i<rowLen;i++)
               d[i] = srcP[f[i]];
         }
      }
   }
};

Layer *Layer::createReorg(int inStride)
{
   return new Reorg(inStride);
}

} // end namespace numerix
