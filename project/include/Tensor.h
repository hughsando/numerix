#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include <vector>
#include <algorithm>
#include <string>
#include <memory.h>

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
   actLeaky,
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
      Tensor *cropAndScale(int inWidth, int inHeight, Tensor *inBuffer = 0);

      int    getIntAt(int inIndex);
      double getFloatAt(int inIndex);

      double getMin();
      double getMax();

      void checkData();
      Tensor *incRef();
      int addRef();
      int decRef();

      static Tensor *makeBuffer(Tensor *inBuffer, int inW, int inH, int inChannels, int inType);
      static unsigned char *allocData(unsigned int inLength);
      static void freeData(u8 *data);

   protected:
      void printSub(const std::string &indent, int offset, int dim, int inMaxElems);
      void updateStrides();
      int refCount;
      ~Tensor();
};


#undef min
#undef max

struct BBox
{
   float x;
   float y;
   float w;
   float h;
   float prob;
   int   classId;

   inline bool operator<(const BBox &inRight) const
   {
      return prob > inRight.prob;
   }

   inline bool overlaps(const BBox &o) const
   {
      float intX = std::min(x+w, o.x+o.w) - std::max(x,o.x);
      if (intX<=0)
         return false;
      float intY = std::min(y+h, o.y+o.h) - std::max(y,o.y);
      if (intY<=0)
         return false;
      float intersect = intX*intY;
      float total = w*h + o.w*o.h - intersect;
      return (intersect > total * 0.5);
   }
};

typedef std::vector<BBox> Boxes;



class Layer
{
   volatile int jobId;
   std::vector<unsigned char *> buffers;

public:
   typedef unsigned char u8;

   static Layer *createConv2D(int inStrideY, int inStrideX,
                              Activation activation, Padding padding,
                              Tensor *weights, Tensor *pweights, Tensor *bias);

   static Layer *createMaxPool(int inSizeX, int inSizeY,
                               int inStrideY, int inStrideX,
                               Padding padding);

   static Layer *createConcat();

   static Layer *createPack(int inStride);

   static Layer *createYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount);


   virtual ~Layer();

   virtual void setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars) { }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer) { return 0; }
   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer) { return 0; }

   virtual void runThread(int inThreadId) { }

   virtual void getBoxes(Boxes &outBoxes) { }

   int getNextJob();

   float *allocFloats(int count,bool inZero=false);
   void   releaseFloats();


   void runThreaded();
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


#endif
