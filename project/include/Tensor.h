#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include <vector>
#include <string>

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

enum Activation
{
   actLinear,
   actRelu,
   actSigmoid,
};

enum Padding
{
   padSame,
   padValid,
};


typedef std::vector<int> Shape;
typedef const std::vector<int> &CShape;

void TensorThrow(const char *err);

class Tensor
{
   public:
      typedef unsigned char u8;

      u8    *data;
      Shape shape;
      Shape strides;
      int   type;
      int   elementSize;
      unsigned int elementCount;

      Tensor(int inType, const Shape &inShape);

      void print(int inMaxElems);
      void fill(int inType, const u8 *inData, int inOffsetElem,  unsigned int inCount);
      void zero(int inOffsetElem, unsigned int inCount);
      void setInt32(int inValue, int inOffsetElem, unsigned int inCount);
      void setFloat64(double inValue, int inOffsetElem, unsigned int inCount);

      void setFlat();
      void setShape(CShape inShape);
      Tensor *reorder(const std::vector<int> &order);

      int    getIntAt(int inIndex);
      double getFloatAt(int inIndex);

      double getMin();
      double getMax();

      void checkData();
      Tensor *incRef();
      int addRef();
      int decRef();

      static unsigned char *allocData(unsigned int inLength);
      static void freeData(u8 *data);

   protected:
      void printSub(const std::string &indent, int offset, int dim, int inMaxElems);
      void updateStrides();
      int refCount;
      ~Tensor();
};

class Layer
{
public:
   static Layer *createConv2D(int inStrideY, int inStrideX,
                              Activation activation, Padding padding,
                              Tensor *weights, Tensor *pweights, Tensor *bias);

   virtual ~Layer() { };

   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer) = 0;
};


#endif
