#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

template<EltwiseOp OP>
class Eltwise : public Layer
{
   int destW;
   int destH;
   int channels;
   int cropIndex;
   int cropX;
   int cropY;

public:
   Eltwise( )
   {
      cropIndex = -1;
      cropX = 0;
      cropY = 0;
   }

   void setCropIndex(int inIndex, int inDx, int inDy)
   {
      cropIndex = inIndex;
      cropX = inDx;
      cropY = inDy;
   }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("Eltwise only supports Float32 tensors");

      if (inSrc0->type != inSrc1->type)
         TensorThrow("Eltwise - input types must match");

      CShape sin0 = inSrc0->shape;
      CShape sin1 = inSrc1->shape;
      if (sin0.size()!=3 || sin1.size()!=3)
         TensorThrow("Eltwise only supports H*W*C tensors");

      bool sizeMismatch = sin0[2]!=sin1[2];
      if (cropIndex>=0)
      {
         CShape dest = cropIndex==0 ? sin0 : sin1;
         CShape other = cropIndex==0 ? sin1 : sin0;
         if (other[0]+cropY < dest[0] || other[1]+cropX < dest[1])
         {
            /*
            printf("Crop index %d\n", cropIndex);
            printf(" sin0 %d %d\n", sin0[0], sin0[1]);
            printf(" sin1 %d %d\n", sin1[0], sin1[1]);
            printf("%d : %d,   %d : %d\n",other[0]+cropY, dest[0], other[1]+cropX, dest[1]);
            */
            sizeMismatch = true;
         }
      }

      if (sizeMismatch)
      {
         char buf[1000];
         sprintf(buf, "Eltwise - mismatch image sizes %dx%dx%d + %dx%dx%d",
                 sin0[0], sin0[1],sin0[2], sin1[0], sin1[1], sin1[2] );
         TensorThrow(buf);
      }

      if (cropIndex==0)
      {
         destH = sin0[0];
         destW = sin0[1];
      }
      else
      {
         destH = sin1[0];
         destW = sin1[1];
      }

      channels = sin0[2];

      startRun();
      Tensor *result = Tensor::makeBuffer(inBuffer, destW, destH, channels, inSrc0->type);

      src0 = inSrc0;
      src1 = inSrc1;
      destTensor = result;

         // Prep in single-thread mode
      src0->cpuRead();
      src1->cpuRead();
      destTensor->cpuWrite();

      runThreaded();

      src0 = 0;
      src1 = 0;
      destTensor = 0;

      endRun();

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
      int nElems = destW * channels;
      int dx0 = 0;
      int dy0 = 0;
      int dx1 = 0;
      int dy1 = 0;
      if (cropIndex==0)
      {
         dx1 = cropX * channels;
         dy1 = cropY;
      }
      else if (cropIndex=1)
      {
         dx0 = cropX * channels;
         dy0 = cropY;
      }

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         const float *s0 = (const float *)src0->cpuRead() + src0Stride[0] * (y+dy0) + dx0;
         const float *s1 = (const float *)src1->cpuRead() + src1Stride[0] * (y+dy1) + dx1;
         float        *d = (float *)destTensor->cpuWrite() + destStride[0] * y;

         for(int x=0;x<nElems;x++)
         {
            if (OP==EltProduct)
               d[x] = s0[x] * s1[x];
            else if (OP==EltSum)
               d[x] = s0[x] + s1[x];
            else if (OP==EltMax)
               d[x] = std::max(s0[x],s1[x]);
         }
      }
   }
};

Layer *Layer::createEltwise(EltwiseOp inOperation)
{
   switch(inOperation)
   {
      case EltProduct:
         return new Eltwise<EltProduct>();
      case EltSum:
         return new Eltwise<EltSum>();
      case EltMax:
         return new Eltwise<EltMax>();
   }

   TensorThrow("Eltwise - unknown operation");
   return 0;
}


} // end namespace numerix

