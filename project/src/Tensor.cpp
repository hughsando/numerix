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

   bool dotdotdot = false;
   int size = shape[dim];
   if (elems<2)
      elems = 2;
   if (size<elems)
      elems = size;
   else if (size>elems)
   {
      dotdotdot = true;
      elems--;
   }

   if (shape.size()==dim+1)
   {
      printf("%s[ ", indent.c_str());
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
         printf("%s   ...\n", indent.c_str());
         printSub(next, offset + (size-2)*strides[dim], dim+1,inMaxElems);
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



template<typename DATA>
void TReorder(const DATA *src, void *inDest,
              const std::vector<int> &shape,
              const std::vector<int> &strides,
              const std::vector<int> &order)
{
   DATA *dest = (DATA *)inDest;
   if (order.size()<2)
   {
      memcpy(dest, src, sizeof(DATA)*shape[0]);
      return;
   }


   int len4    = order.size() > 4 ? shape[ order[4] ] : 1;
   int stride4 = order.size() > 4 ? strides[ order[4] ] : 0;
   int len3    = order.size() > 3 ? shape[ order[3] ] : 1;
   int stride3 = order.size() > 3 ? strides[ order[3] ] : 0;
   int len2    = order.size() > 2 ? shape[ order[2] ] : 1;
   int stride2 = order.size() > 2 ? strides[ order[2] ] : 0;
   int len1    = order.size() > 1 ? shape[ order[1] ] : 1;
   int stride1 = order.size() > 1 ? strides[ order[1] ] : 0;
   int len0    = shape[ order[0] ];
   int stride0 = strides[ order[0] ];

   int src4 = 0;
   for(int i4=0;i4<len4;i4++)
   {
      int src3 = src4;
      src4 += stride4;
      for(int i3=0;i3<len3;i3++)
      {
         int src2 = src3;
         src3 += stride3;
         for(int i2=0;i2<len2;i2++)
         {
            int src1 = src2;
            src2 += stride2;
            for(int i1=0;i1<len1;i1++)
            {
               int src0 = src1;
               src1 += stride1;
               for(int i0=0;i0<len0;i0++)
               {
                  *dest++ = src[src0];
                  src0 += stride0;
               }
            }
         }
      }
   }
}




Tensor *Tensor::reorder(const std::vector<int> &order)
{
   if (order.size()!=shape.size())
      TensorThrow( "Wrong number of coordinate in reorer" );

   for(int i=0;i<order.size();i++)
      if (std::find(order.begin(), order.end(), i) == order.end())
         TensorThrow( "Missing coordinate in reorder" );

   if (order.size()>5)
      TensorThrow( "reorder - not implemented with this many coordinates" );

   std::vector<int> targetShape(order.size());
   for(int i=0;i<order.size();i++)
      targetShape[i] = shape[ order[i] ];
   Tensor *t = new Tensor(type, targetShape);
   void *d = t->data;
   switch(type)
   {
      case Float32: TReorder(((float *)data)         , d, shape, strides, order ); break;
      case Float64: TReorder(((double *)data)        , d, shape, strides, order ); break;
      case UInt8:   TReorder(((unsigned char *)data) , d, shape, strides, order ); break;
      case UInt16:  TReorder(((unsigned short *)data), d, shape, strides, order ); break;
      case UInt32:  TReorder(((unsigned int *)data)  , d, shape, strides, order ); break;
      case UInt64:  TReorder(((TUInt64 *)data)       , d, shape, strides, order ); break;
      case Int8:    TReorder(((signed char *)data)   , d, shape, strides, order ); break;
      case Int16:   TReorder(((short *)data)         , d, shape, strides, order ); break;
      case Int32:   TReorder(((int *)data)           , d, shape, strides, order ); break;
      case Int64:   TReorder(((TInt64 *)data)        , d, shape, strides, order ); break;
   }

   return t;
}


