#include <Tensor.h>

#include <stdexcept>

#define LOW_SENTINEL  67
#define HIGH_SENTINEL 68

unsigned char *Tensor::allocData(unsigned int inLength)
{
   // Allocate 6 bytes at beginning, plus up to and extra 12 to ensure 16-byte alignment, and
   //  1 at the end
   // Add 3 and trunc to ensure multiple of 4
   u8 *alloc = new u8[ (inLength+6+12+1+3) & ~3];
   int offset16 = 16 - (((int)(size_t)alloc) & 0xf);
   if (offset16<6)
      offset16 += 16;

   u8 *result = alloc + offset16;
   result[-1] = LOW_SENTINEL;
   result[inLength] = HIGH_SENTINEL;

   result[-2] = offset16;
   *(unsigned int *)(result-6) = inLength;
   return result;
}

void Tensor::freeData(unsigned char *data)
{
   if (data)
   {
      int align16 = data[-2];
      u8 *ptr = data - align16;
      delete [] ptr;
   }
}


Tensor::Tensor( int inType, const Shape &inShape )
   : data(0), type(inType), shape(inShape)
{
   refCount = 1;
   elementCount = 1;
   if (shape.size()==0)
      shape.push_back(1);
   else
      for(int i=0;i<shape.size();i++)
         elementCount *= shape[i];

   int nType = inType & NumberMask;
   if (nType==SignedInteger || nType==UnsignedInteger || nType==Floating)
   {
      elementSize = (nType & BitsMask);

      data = allocData( elementCount * elementSize );
   }
   else
      throw std::logic_error("bad tensor type");
}



void Tensor::fill(int inType, const u8 *inData, unsigned int inLength)
{
   if (inType==type)
   {
      int len = elementCount < inLength ? elementCount : inLength;
      memcpy(data, inData, len * elementSize);
   }
}



void Tensor::checkData()
{
   if (!data)
      throw std::logic_error("tensor missing data");

   if (data[-1]!=LOW_SENTINEL)
      throw std::logic_error("tensor underwrite");

   unsigned int length = *(int *)(data-6);
   if (data[length] != HIGH_SENTINEL)
      throw std::logic_error("tensor overwrite");
}

Tensor::~Tensor()
{
   if (data)
      checkData();
   freeData(data);
}

int Tensor::addRef()
{
   // TODO - atomic
   refCount++;
   return refCount;
}

int Tensor::decRef()
{
   // TODO - atomic
   refCount--;
   if (refCount<=0)
   {
      delete this;
      return 0;
   }
   return refCount;
}


