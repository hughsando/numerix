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
   {
      shape.push_back(1);
      strides.push_back(1);
   }
   else
   {
      strides.resize(shape.size());
      for(int i=shape.size()-1;i>=0;i--)
      {
         strides[i] = elementCount;
         elementCount *= shape[i];
      }
   }

   int nType = inType & NumberMask;
   if (nType==SignedInteger || nType==UnsignedInteger || nType==Floating)
   {
      elementSize = (inType & BitsMask)>>3;

      data = allocData( elementCount * elementSize );
   }
   else
      throw std::logic_error("bad tensor type");
}

void Tensor::printSub(const std::string &indent, int offset, int dim, int inMaxElems)
{
   int elems = inMaxElems - (shape.size() - dim - 1) * 4;
   if (elems<2)
      elems = 2;

   bool dotdotdot = false;
   int size = shape[dim];
   if (size<elems)
      elems = size;
   else if (size>elems)
   {
      dotdotdot = true;
      elems = size-1;
   }

   if (shape.size()==dim+1)
   {
      printf("%s[ ",indent.c_str());
      if (!(type & Floating))
      {
         for(int i=0;i<elems;i++)
            printf("%d ", getIntAt(offset + i));
         if (dotdotdot)
         {
            printf("... ");
            printf("%d", getIntAt(offset +  size-1) );
         }
      }
      else
      {
         for(int i=0;i<elems;i++)
            printf("%f ", getFloatAt(offset + i));
         if (dotdotdot)
         {
            printf("... ");
            printf("%f", getFloatAt(offset +  size-1) );
         }
      }
      printf("]\n");
   }
   else
   {
      std::string next = indent + "  ";
      printf("%s[\n",indent.c_str());
      for(int i=0;i<elems;i++)
         printSub(next, offset + i*strides[dim], dim+1,inMaxElems);

      if (dotdotdot)
      {
         printf("... ");
         printSub(next, offset + (size-1)*strides[dim], shape[dim]-1,inMaxElems);
      }

      printf("%s]\n",indent.c_str());
   }
}

void Tensor::print(int inMaxElems)
{
   std::string indent = "";
   printSub(indent, 0, 0, inMaxElems);
}


int Tensor::getIntAt(int inIndex)
{
   switch(type)
   {
      case Float32: return ((float *)data)[inIndex];
      case Float64: return ((double *)data)[inIndex];
      case UInt8: return ((unsigned char *)data)[inIndex];
      case UInt16: return ((unsigned short *)data)[inIndex];
      case UInt32: return ((unsigned int *)data)[inIndex];
      case UInt64: return ((TUInt64 *)data)[inIndex];
      case Int8: return ((signed char *)data)[inIndex];
      case Int16: return ((short *)data)[inIndex];
      case Int32: return ((int *)data)[inIndex];
      case Int64: return ((TInt64 *)data)[inIndex];
   }
   return 0;
}

double Tensor::getFloatAt(int inIndex)
{
   switch(type)
   {
      case Float32: return ((float *)data)[inIndex];
      case Float64: return ((double *)data)[inIndex];
      case UInt8: return ((unsigned char *)data)[inIndex];
      case UInt16: return ((unsigned short *)data)[inIndex];
      case UInt32: return ((unsigned int *)data)[inIndex];
      case UInt64: return ((TUInt64 *)data)[inIndex];
      case Int8: return ((signed char *)data)[inIndex];
      case Int16: return ((short *)data)[inIndex];
      case Int32: return ((int *)data)[inIndex];
      case Int64: return ((TInt64 *)data)[inIndex];
   }
   return 0;
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



template<typename DEST,typename SRC>
void TTFill(DEST *dest,const SRC *src, unsigned int inCount)
{
   for(int i=0;i<inCount;i++)
      dest[i] = src[i];
}

template<typename T>
void TFill(T *outData, int inType, const unsigned char *inData, unsigned int inCount)
{
   switch(inType)
   {
      case Float32: TTFill( outData, ((float *)inData), inCount); break;
      case Float64: TTFill( outData, ((double *)inData), inCount); break;
      case UInt8:   TTFill( outData, ((unsigned char *)inData), inCount); break;
      case UInt16:  TTFill( outData, ((unsigned short *)inData), inCount); break;
      case UInt32:  TTFill( outData, ((unsigned int *)inData), inCount); break;
      case UInt64:  TTFill( outData, ((TUInt64 *)inData), inCount); break;
      case Int8:    TTFill( outData, ((signed char *)inData), inCount); break;
      case Int16:   TTFill( outData, ((short *)inData), inCount); break;
      case Int32:   TTFill( outData, ((int *)inData), inCount); break;
      case Int64:   TTFill( outData, ((TInt64 *)inData), inCount); break;
   }

}


void Tensor::fill(int inType, const unsigned char *inData, int inOffsetElem,  unsigned int inCount)
{
   if (inType==type)
   {
      memcpy(data + inOffsetElem*elementSize, inData, inCount*elementSize);
   }
   else
   {
      switch(type)
      {
         case Float32: TFill( ((float *)data) + inOffsetElem, inType, inData, inCount); break;
         case Float64: TFill( ((double *)data) + inOffsetElem, inType, inData, inCount); break;
         case UInt8:   TFill( ((unsigned char *)data) + inOffsetElem, inType, inData, inCount); break;
         case UInt16:  TFill( ((unsigned short *)data) + inOffsetElem, inType, inData, inCount); break;
         case UInt32:  TFill( ((unsigned int *)data) + inOffsetElem, inType, inData, inCount); break;
         case UInt64:  TFill( ((TUInt64 *)data) + inOffsetElem, inType, inData, inCount); break;
         case Int8:    TFill( ((signed char *)data) + inOffsetElem, inType, inData, inCount); break;
         case Int16:   TFill( ((short *)data) + inOffsetElem, inType, inData, inCount); break;
         case Int32:   TFill( ((int *)data) + inOffsetElem, inType, inData, inCount); break;
         case Int64:   TFill( ((TInt64 *)data) + inOffsetElem, inType, inData, inCount); break;
      }

   }
}

void Tensor::zero(int inOffsetElem, unsigned int inCount)
{
   memset( data + inOffsetElem*elementSize, 0, inCount*elementSize );
}



template<typename SRC,typename DEST>
void TSet(SRC inValue, DEST *dest, unsigned int inCount)
{
   for(int i=0;i<inCount;i++)
      dest[i] = inValue;
}

void Tensor::setInt32(int inValue, int inOffsetElem, unsigned int inCount)
{
   if (inValue==0)
      zero(inOffsetElem,inCount);
   else
   {
      switch(type)
      {
         case Float32: TSet(inValue, ((float *)data) + inOffsetElem, inCount); break;
         case Float64: TSet(inValue, ((double *)data) + inOffsetElem, inCount); break;
         case UInt8:   TSet(inValue, ((unsigned char *)data) + inOffsetElem, inCount); break;
         case UInt16:  TSet(inValue, ((unsigned short *)data) + inOffsetElem, inCount); break;
         case UInt32:  TSet(inValue, ((unsigned int *)data) + inOffsetElem, inCount); break;
         case UInt64:  TSet(inValue, ((TUInt64 *)data) + inOffsetElem, inCount); break;
         case Int8:    TSet(inValue, ((signed char *)data) + inOffsetElem, inCount); break;
         case Int16:   TSet(inValue, ((short *)data) + inOffsetElem, inCount); break;
         case Int32:   TSet(inValue, ((int *)data) + inOffsetElem, inCount); break;
         case Int64:   TSet(inValue, ((TInt64 *)data) + inOffsetElem, inCount); break;
      }
   }
}

void Tensor::setFloat64(double inValue, int inOffsetElem, unsigned int inCount)
{
   if (inValue==0)
      zero(inOffsetElem,inCount);
   else
   {
      switch(type)
      {
         case Float32: TSet(inValue, ((float *)data) + inOffsetElem, inCount); break;
         case Float64: TSet(inValue, ((double *)data) + inOffsetElem, inCount); break;
         case UInt8:   TSet(inValue, ((unsigned char *)data) + inOffsetElem, inCount); break;
         case UInt16:  TSet(inValue, ((unsigned short *)data) + inOffsetElem, inCount); break;
         case UInt32:  TSet(inValue, ((unsigned int *)data) + inOffsetElem, inCount); break;
         case UInt64:  TSet(inValue, ((TUInt64 *)data) + inOffsetElem, inCount); break;
         case Int8:    TSet(inValue, ((signed char *)data) + inOffsetElem, inCount); break;
         case Int16:   TSet(inValue, ((short *)data) + inOffsetElem, inCount); break;
         case Int32:   TSet(inValue, ((int *)data) + inOffsetElem, inCount); break;
         case Int64:   TSet(inValue, ((TInt64 *)data) + inOffsetElem, inCount); break;
      }
   }
}

