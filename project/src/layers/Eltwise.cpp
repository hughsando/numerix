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
   int srcW;
   int srcH;
   int channels;

public:
   Eltwise( )
   {
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

      if (sin0[0]!=sin1[0] || sin0[1]!=sin1[1] || sin0[2]!=sin1[2])
      {
         char buf[1000];
         sprintf(buf, "Eltwise - mismatch image sizes %dx%dx%d + %dx%dx%d",
                 sin0[0], sin0[1],sin0[2], sin1[0], sin1[1], sin1[2] );
         TensorThrow(buf);
      }

      srcH = sin0[0];
      srcW = sin0[1];
      channels = sin0[2];

      startRun();
      Tensor *result = Tensor::makeBuffer(inBuffer, srcW, srcH, channels, inSrc0->type);

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
      int nElems = srcW * channels;

      while(true)
      {
         int y = getNextJob();
         if (y>=srcH)
            break;

         const float *s0 = (const float *)src0->cpuRead() + src0Stride[0] * y;
         const float *s1 = (const float *)src1->cpuRead() + src1Stride[0] * y;
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

