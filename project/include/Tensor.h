#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include <vector>

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

class Tensor
{
   public:
      typedef unsigned char u8;

      u8    *data;
      Shape shape;
      int   type;
      int   elementSize;
      unsigned int elementCount;

      Tensor(int inType, const Shape &inShape);

      void fill(int inType, const u8 *inData, unsigned int inCount);

      void checkData();
      int addRef();
      int decRef();

      static unsigned char *allocData(unsigned int inLength);
      static void freeData(u8 *data);

   protected:
      int refCount;
      ~Tensor();
};


#endif
