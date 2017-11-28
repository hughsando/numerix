#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include <vector>
#include <algorithm>
#include <string>
#include <memory.h>



namespace numerix
{

class Tensor;

enum Activation
{
   actLinear,
   actRelu,
   actSigmoid,
   actLeaky,
};

enum Padding
{
   padSame,
   padValid,
};

}// end namespace numerix


#ifdef NX_GPU
#include "NxGpu.h"
#endif


namespace numerix
{

#ifdef _WIN32
typedef __int64 TInt64;
typedef unsigned __int64 TUInt64;
// TODO - EMSCRIPTEN?
#else
typedef int64_t TInt64;
typedef uint64_t TUInt64;
#endif



enum DataType
{
   SignedInteger   = 0x01000000,
   UnsignedInteger = 0x02000000,
   Floating        = 0x04000000,
   AsciiString     = 0x08000000,

   BitsMask        = 0x00ffffff,
   NumberMask      = 0x07000000,

   Float32        = Floating | 32,
   Float64        = Floating | 64,
   UInt8          = UnsignedInteger | 8,
   UInt16         = UnsignedInteger | 16,
   UInt32         = UnsignedInteger | 32,
   UInt64         = UnsignedInteger | 64,
   Int8           = SignedInteger | 8,
   Int16          = SignedInteger | 16,
   Int32          = SignedInteger | 32,
   Int64          = SignedInteger | 64,
};


typedef std::vector<int> Shape;
typedef const std::vector<int> &CShape;


void TensorThrow(const char *err);

typedef unsigned char u8;

class TensorData
{
   u8  *cpu;
   #ifdef NX_GPU
   GpuData *gpu;
   bool cpuValid;
   bool gpuValid;
   bool cpuNchw;
   bool gpuNchw;
   #endif
   int size;

 public:
   inline TensorData(int inSize) :
       size(inSize), cpu(0), refCount(1)
   {
      #ifdef NX_GPU
      gpu = 0;
      cpuValid = true;
      gpuValid = false;
      cpuNchw = false;
      gpuNchw = false;
      #else
      cpu = allocCpuAligned(size);
      #endif
   }

   #ifdef NX_GPU
   bool isGpuNchw()
   {
      if (gpuValid)
         return gpuNchw;
      return cpuNchw;
   }

   inline u8 *getCpu(bool updateCpu, bool invalidateGpu, bool inNchw, Tensor *inTensor)
   {
      if (!cpu)
         cpu = allocCpuAligned(size);

      if (updateCpu && !cpuValid && gpuValid && gpu)
      {
         if (inNchw!=gpuNchw)
            gpuDownloadConvert(cpu, gpu, size, inNchw, inTensor);
         else
            gpuDownload(cpu, gpu, size);
      }
      cpuNchw = inNchw;

      cpuValid = true;
      if (invalidateGpu)
         gpuValid = false;

      return cpu;
   }

   inline u8 *getGpu(bool updateGpu,bool invalidateCpu, bool inNchw, Tensor *inTensor)
   {
      if (!gpu)
         gpu = gpuAlloc(size);
      if (updateGpu && !gpuValid && cpu && cpuValid)
      {
         if (inNchw!=cpuNchw)
            gpuUploadConvert(gpu, cpu, size, inNchw, inTensor);
         else
            gpuUpload(gpu, cpu, size);
      }
      gpuValid = true;
      if (invalidateCpu)
         cpuValid = false;
      gpuNchw = inNchw;
      return (u8 *)gpu;
   }

   #else
   inline u8 *getCpu()
   {
      return cpu;
   }
   #endif

   static u8 *allocCpuAligned(int inSize);
   static void freeCpuAligned(void *inPtr);

   void decRef();
   TensorData *incRef();
   void check();

 protected:
   volatile int refCount;
   ~TensorData();
   TensorData(const TensorData &);
   void operator =(const TensorData &);
};

class Tensor
{
   public:
      Shape shape;
      Shape strides;
      int   type;
      int   elementSize;
      unsigned int elementCount;

      Tensor(int inType, const Shape &inShape);

      #ifdef NX_GPU
      const inline bool isGpuNchw() { return data->isGpuNchw(); }
      const inline u8 *cpuRead(bool inNchw=false) { return data->getCpu(true,false,inNchw,this); }
      inline       u8 *cpuWrite(bool inNchw=false) { return data->getCpu(false,true,inNchw,this); }
      inline       u8 *cpuWritePart(bool inNchw=false) { return data->getCpu(true,true,inNchw,this); }

      const inline u8 *gpuRead(bool inNchw=false) { return data->getGpu(true,false,inNchw,this); }
      inline       u8 *gpuWrite(bool inNchw=false) { return data->getGpu(false,true,inNchw,this); }
      #else
      const inline bool isGpuNchw() { return false; }
      const inline u8 *cpuRead(bool x=false) { return data->getCpu(); }
      inline       u8 *cpuWrite(bool x=false) { return data->getCpu(); }
      inline       u8 *cpuWritePart(bool x=false) { return data->getCpu(); }

      const inline u8 *gpuRead(bool x=false) { return 0; }
      inline       u8 *gpuWrite(bool x=false) { return 0; }
      #endif

      void print(int inMaxElems);
      void fill(int inType, const u8 *inData, int inOffsetElem,  unsigned int inCount);
      void zero(int inOffsetElem, unsigned int inCount);
      void setInt32(int inValue, int inOffsetElem, unsigned int inCount);
      void setFloat64(double inValue, int inOffsetElem, unsigned int inCount);

      void setFlat();
      void setShape(CShape inShape);
      Tensor *reorder(const std::vector<int> &order);
      Tensor *cropAndScale(int inWidth, int inHeight, Tensor *inBuffer = 0);

      int   getByteCount() const { return elementCount * elementSize; }

      int    getIntAt(int inIndex);
      double getFloatAt(int inIndex);

      double getMin();
      double getMax();

      Tensor *incRef();
      int addRef();
      int decRef();

      static Tensor *makeBuffer(Tensor *inBuffer, int inW, int inH, int inChannels, int inType);

      void convertToNchw(u8 *outData, const u8 *inData) const;
      void convertToNhwc(u8 *outData, const u8 *inData) const;

   protected:
      TensorData  *data;
      void printSub(const std::string &indent, int offset, int dim, int inMaxElems);
      void updateStrides();
      volatile int refCount;
      ~Tensor();
};

   

struct Shape1 : public std::vector<int>
{
   inline Shape1(int inS0) : std::vector<int>(1)
   {
      (*this)[0] = inS0;
   }
};
struct Shape2 : public std::vector<int>
{
   inline Shape2(int inS0,int inS1) : std::vector<int>(2)
   {
      (*this)[0] = inS0;
      (*this)[1] = inS1;
   }
};
struct Shape3 : public std::vector<int>
{
   inline Shape3(int inS0,int inS1,int inS2) : std::vector<int>(3)
   {
      (*this)[0] = inS0;
      (*this)[1] = inS1;
      (*this)[2] = inS2;
   }
};
struct Shape4 : public std::vector<int>
{
   inline Shape4(int inS0,int inS1,int inS2, int inS3) : std::vector<int>(4)
   {
      (*this)[0] = inS0;
      (*this)[1] = inS1;
      (*this)[2] = inS2;
      (*this)[3] = inS3;
   }
};

} // end namespace numerix


#endif
