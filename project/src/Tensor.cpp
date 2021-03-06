#include <Tensor.h>
#include <hx/Thread.h>

#include <stdexcept>

#undef min
#undef max

#define LOW_SENTINEL  67
#define HIGH_SENTINEL 68

namespace numerix
{


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
      updateStrides();
   }

   int nType = inType & NumberMask;
   if (nType==SignedInteger || nType==UnsignedInteger || nType==Floating)
   {
      elementSize = (inType & BitsMask)>>3;

      data = new TensorData( elementCount * elementSize );
   }
   else
      throw std::logic_error("bad tensor type");
}

void Tensor::updateStrides()
{
   elementCount = 1;
   strides.resize(shape.size());
   for(int i=shape.size()-1;i>=0;i--)
   {
      strides[i] = elementCount;
      elementCount *= shape[i];
   }
}

Tensor *Tensor::makeBuffer(Tensor *inBuffer, int inW, int inH, int inChannels, int inType)
{
   bool match = false;
   if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==inType)
   {
      CShape s = inBuffer->shape;
      if (s[0]==inH && s[1]==inW && s[2]==inChannels)
         return inBuffer;
   }

   Shape s(3);
   s[0] = inH;
   s[1] = inW;
   s[2] = inChannels;
   return new Tensor( inType, s );
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
            printf("%g ", getFloatAt(offset + i));
         if (dotdotdot)
         {
            printf("... ");
            printf("%g", getFloatAt(offset +  size-1) );
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

         printSub(next, offset + (size-1)*strides[dim], dim+1,inMaxElems);
      }

      printf("%s]\n",indent.c_str());
   }
}

void Tensor::BoundsError(int inDim, int inValue)
{
   char buf[1000];
   sprintf(buf,"BoundsError - component %d - requested element %d of %d",inDim, inValue, shape[inDim]);
   TensorThrow(buf);
}

void Tensor::ShapeError(int inRequested)
{
   char buf[1000];
   sprintf(buf,"ShapeError requested shape %d, but has %d",inRequested, (int)shape.size());
   TensorThrow(buf);
}


void Tensor::print(int inMaxElems)
{
   std::string indent = "";
   printSub(indent, 0, 0, inMaxElems);
}


int Tensor::getIntAt(int inIndex)
{
   const void *d = cpuRead();
   switch(type)
   {
      case Float32: return ((float *)d)[inIndex];
      case Float64: return ((double *)d)[inIndex];
      case UInt8: return ((unsigned char *)d)[inIndex];
      case UInt16: return ((unsigned short *)d)[inIndex];
      case UInt32: return ((unsigned int *)d)[inIndex];
      case UInt64: return ((TUInt64 *)d)[inIndex];
      case Int8: return ((signed char *)d)[inIndex];
      case Int16: return ((short *)d)[inIndex];
      case Int32: return ((int *)d)[inIndex];
      case Int64: return ((TInt64 *)d)[inIndex];
   }
   return 0;
}

double Tensor::getFloatAt(int inIndex)
{
   const void *d = cpuRead();
   switch(type)
   {
      case Float32: return ((float *)d)[inIndex];
      case Float64: return ((double *)d)[inIndex];
      case UInt8: return ((unsigned char *)d)[inIndex];
      case UInt16: return ((unsigned short *)d)[inIndex];
      case UInt32: return ((unsigned int *)d)[inIndex];
      case UInt64: return ((TUInt64 *)d)[inIndex];
      case Int8: return ((signed char *)d)[inIndex];
      case Int16: return ((short *)d)[inIndex];
      case Int32: return ((int *)d)[inIndex];
      case Int64: return ((TInt64 *)d)[inIndex];
   }
   return 0;
}




Tensor::~Tensor()
{
   if (data)
      data->decRef();
}


int Tensor::addRef()
{
   int now = HxAtomicInc(&refCount) + 1;
   return now;
}

Tensor *Tensor::incRef()
{
   addRef();
   return this;
}

int Tensor::decRef()
{
   // TODO - atomic
   int now = HxAtomicDec(&refCount) - 1;
   if (now<=0)
   {
      delete this;
      return 0;
   }
   return now;
}


void Tensor::setFlat()
{
   shape = Shape();
   shape.push_back(elementCount);
   updateStrides();
}

void Tensor::setShape(CShape inShape)
{
   if (inShape.empty())
      setFlat();
   else
   {
      int n = inShape[0];

      for(int i=1;i<inShape.size();i++)
         n *= inShape[i];
      if (n!=elementCount)
         TensorThrow("setShape - sizes do not match");

      shape = inShape;
      updateStrides();
   }
}


Tensor *Tensor::resizeAxis(int inAxis, int inNewSize, int inDataPos)
{
   Shape s = shape;
   if (inAxis<0 || s.size()<inAxis)
      TensorThrow("resizeAxis - axis out of bounds");

   int srcSize = shape[inAxis];
   s[inAxis] = inNewSize;

   Tensor *result = new Tensor( type, s );

   int chunkCount = 1;
   for(int i=0;i<inAxis;i++)
      chunkCount *= s[i];

   int srcChunkSize = elementSize;
   int destChunkSize = elementSize;
   for(int i=inAxis;i<s.size();i++)
   {
      srcChunkSize *= shape[i];
      destChunkSize *= s[i];
   }

   const u8 *src = cpuRead();

   int chunkSize = std::min(srcChunkSize, destChunkSize);
   int leadingZeros = 0;
   int trailingZeros = 0;

   // ...leading zero...
   // data
   // ...trailing zero...

   if (srcSize<inNewSize)
   {
      int s0 = inDataPos;
      if (s0<0)
         s0 = 0;
      else if (s0 > inNewSize-srcSize)
         s0 = inNewSize-srcSize;

      leadingZeros = s0 * strides[inAxis] * elementSize;
      trailingZeros = (inNewSize-srcSize-s0) * strides[inAxis] * elementSize;
   }
   else if (inNewSize<srcSize)
   {
      int s0 = inDataPos;
      if (s0<0)
         s0 = 0;
      else if (s0+inNewSize>srcSize)
         s0 = srcSize-inNewSize;

      src += s0 * strides[inAxis] * elementSize;
   }

   u8 *dest = result->cpuWrite();

   for(int c=0; c<chunkCount;c++)
   {
      if (leadingZeros)
         memset(dest, 0, leadingZeros);

      memcpy(dest+leadingZeros, src, chunkSize);

      if (trailingZeros)
         memset(dest+leadingZeros+chunkSize, 0, trailingZeros);

      src += srcChunkSize;
      dest += destChunkSize;
   }

   return result;
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
   bool all = inCount == elementCount;
   u8 *d = all ? cpuWrite() : cpuWritePart();

   if (inType==type)
   {
      memcpy(d + inOffsetElem*elementSize, inData, inCount*elementSize);
   }
   else
   {
      switch(type)
      {
         case Float32: TFill( ((float *)d) + inOffsetElem, inType, inData, inCount); break;
         case Float64: TFill( ((double *)d) + inOffsetElem, inType, inData, inCount); break;
         case UInt8:   TFill( ((unsigned char *)d) + inOffsetElem, inType, inData, inCount); break;
         case UInt16:  TFill( ((unsigned short *)d) + inOffsetElem, inType, inData, inCount); break;
         case UInt32:  TFill( ((unsigned int *)d) + inOffsetElem, inType, inData, inCount); break;
         case UInt64:  TFill( ((TUInt64 *)d) + inOffsetElem, inType, inData, inCount); break;
         case Int8:    TFill( ((signed char *)d) + inOffsetElem, inType, inData, inCount); break;
         case Int16:   TFill( ((short *)d) + inOffsetElem, inType, inData, inCount); break;
         case Int32:   TFill( ((int *)d) + inOffsetElem, inType, inData, inCount); break;
         case Int64:   TFill( ((TInt64 *)d) + inOffsetElem, inType, inData, inCount); break;
      }

   }
}

void Tensor::zero(int inOffsetElem, unsigned int inCount)
{
   memset( cpuWrite() + inOffsetElem*elementSize, 0, inCount*elementSize );
}

void Tensor::zero()
{
   zero(0,elementCount);
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
      void *d = cpuWritePart();
      switch(type)
      {
         case Float32: TSet(inValue, ((float *)d) + inOffsetElem, inCount); break;
         case Float64: TSet(inValue, ((double *)d) + inOffsetElem, inCount); break;
         case UInt8:   TSet(inValue, ((unsigned char *)d) + inOffsetElem, inCount); break;
         case UInt16:  TSet(inValue, ((unsigned short *)d) + inOffsetElem, inCount); break;
         case UInt32:  TSet(inValue, ((unsigned int *)d) + inOffsetElem, inCount); break;
         case UInt64:  TSet(inValue, ((TUInt64 *)d) + inOffsetElem, inCount); break;
         case Int8:    TSet(inValue, ((signed char *)d) + inOffsetElem, inCount); break;
         case Int16:   TSet(inValue, ((short *)d) + inOffsetElem, inCount); break;
         case Int32:   TSet(inValue, ((int *)d) + inOffsetElem, inCount); break;
         case Int64:   TSet(inValue, ((TInt64 *)d) + inOffsetElem, inCount); break;
      }
   }
}



void Tensor::setFloat64(double inValue, int inOffsetElem, unsigned int inCount)
{
   if (inValue==0)
      zero(inOffsetElem,inCount);
   else
   {
      void *d = cpuWritePart();
      switch(type)
      {
         case Float32: TSet(inValue, ((float *)d) + inOffsetElem, inCount); break;
         case Float64: TSet(inValue, ((double *)d) + inOffsetElem, inCount); break;
         case UInt8:   TSet(inValue, ((unsigned char *)d) + inOffsetElem, inCount); break;
         case UInt16:  TSet(inValue, ((unsigned short *)d) + inOffsetElem, inCount); break;
         case UInt32:  TSet(inValue, ((unsigned int *)d) + inOffsetElem, inCount); break;
         case UInt64:  TSet(inValue, ((TUInt64 *)d) + inOffsetElem, inCount); break;
         case Int8:    TSet(inValue, ((signed char *)d) + inOffsetElem, inCount); break;
         case Int16:   TSet(inValue, ((short *)d) + inOffsetElem, inCount); break;
         case Int32:   TSet(inValue, ((int *)d) + inOffsetElem, inCount); break;
         case Int64:   TSet(inValue, ((TInt64 *)d) + inOffsetElem, inCount); break;
      }
   }
}


void Tensor::setFloatAt(int inIndex,float inValue)
{
   void *d = cpuWritePart();
   switch(type)
   {
      case Float32: ((float *)d)[ inIndex ]= inValue; break;
      case Float64: ((double *)d)[ inIndex ]= inValue; break;
      case UInt8:   ((unsigned char *)d)[ inIndex ]= inValue; break;
      case UInt16:  ((unsigned short *)d)[ inIndex ]= inValue; break;
      case UInt32:  ((unsigned int *)d)[ inIndex ]= inValue; break;
      case UInt64:  ((TUInt64 *)d)[ inIndex ]= inValue; break;
      case Int8:    ((signed char *)d)[ inIndex ]= inValue; break;
      case Int16:   ((short *)d)[ inIndex ]= inValue; break;
      case Int32:   ((int *)d)[ inIndex ]= inValue; break;
      case Int64:   ((TInt64 *)d)[ inIndex ]= inValue; break;
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

   int s = order.size();
   int l = s-1;

   int len4    = s > 4 ? shape[ order[l-4] ] : 1;
   int stride4 = s > 4 ? strides[ order[l-4] ] : 0;
   int len3    = s > 3 ? shape[ order[l-3] ] : 1;
   int stride3 = s > 3 ? strides[ order[l-3] ] : 0;
   int len2    = s > 2 ? shape[ order[l-2] ] : 1;
   int stride2 = s > 2 ? strides[ order[l-2] ] : 0;
   int len1    = s > 1 ? shape[ order[l-1] ] : 1;
   int stride1 = s > 1 ? strides[ order[l-1] ] : 0;
   int len0    = shape[ order[l] ];
   int stride0 = strides[ order[l] ];

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
   {
      char buffer[1000];
      sprintf(buffer,"Wrong number of coordinates in reorder %d!=%d", (int)order.size(),(int)shape.size() );
      TensorThrow(buffer);
   }

   for(int i=0;i<order.size();i++)
      if (std::find(order.begin(), order.end(), i) == order.end())
         TensorThrow( "Missing coordinate in reorder" );

   if (order.size()>5)
      TensorThrow( "reorder - not implemented with this many coordinates" );

   std::vector<int> targetShape(order.size());
   for(int i=0;i<order.size();i++)
      targetShape[i] = shape[ order[i] ];
   Tensor *t = new Tensor(type, targetShape);
   void *d = t->cpuWrite();
   const void *src = cpuRead();
   switch(type)
   {
      case Float32: TReorder(((float *)src)         , d, shape, strides, order ); break;
      case Float64: TReorder(((double *)src)        , d, shape, strides, order ); break;
      case UInt8:   TReorder(((unsigned char *)src) , d, shape, strides, order ); break;
      case UInt16:  TReorder(((unsigned short *)src), d, shape, strides, order ); break;
      case UInt32:  TReorder(((unsigned int *)src)  , d, shape, strides, order ); break;
      case UInt64:  TReorder(((TUInt64 *)src)       , d, shape, strides, order ); break;
      case Int8:    TReorder(((signed char *)src)   , d, shape, strides, order ); break;
      case Int16:   TReorder(((short *)src)         , d, shape, strides, order ); break;
      case Int32:   TReorder(((int *)src)           , d, shape, strides, order ); break;
      case Int64:   TReorder(((TInt64 *)src)        , d, shape, strides, order ); break;
   }

   return t;
}



void Tensor::convertToNchw(u8 *d, const u8 *src) const
{
   int s = shape.size();
   Shape order;
   Shape srcShape = shape;

   if (s==3)
      order = Shape3(2,0,1);
   else if (s==4)
      order = Shape4(0,3,1,2);
   else
      TensorThrow("NCHW conversion - bad size");

   //printf("convertToNchw %dx%dx%dx%d\n", s==4 ? shape[0] : 0, shape[s-3], shape[s-2], shape[s-1] );

   switch(type)
   {
      case Float32: TReorder(((float *)src)         , d, srcShape, strides, order ); break;
      case Float64: TReorder(((double *)src)        , d, srcShape, strides, order ); break;
      case UInt8:   TReorder(((unsigned char *)src) , d, srcShape, strides, order ); break;
      case UInt16:  TReorder(((unsigned short *)src), d, srcShape, strides, order ); break;
      case UInt32:  TReorder(((unsigned int *)src)  , d, srcShape, strides, order ); break;
      case UInt64:  TReorder(((TUInt64 *)src)       , d, srcShape, strides, order ); break;
      case Int8:    TReorder(((signed char *)src)   , d, srcShape, strides, order ); break;
      case Int16:   TReorder(((short *)src)         , d, srcShape, strides, order ); break;
      case Int32:   TReorder(((int *)src)           , d, srcShape, strides, order ); break;
      case Int64:   TReorder(((TInt64 *)src)        , d, srcShape, strides, order ); break;
   }
}

void Tensor::convertToNhwc(u8 *d, const u8 *src) const
{
   int s = shape.size();
   Shape srcShape;
   Shape order;
   if (s==3)
   {
      srcShape = Shape3( shape[2], shape[0], shape[1] );
      order  = Shape3( 1, 2, 0 );
   }
   else if (s==4)
   {
      srcShape = Shape4( shape[0], shape[3], shape[1], shape[2] );
      order  = Shape4( 0, 2, 3, 1 );
   }
   else
      TensorThrow("NHWC conversion - bad size");


   //printf("convertToNhwc %dx%dx%dx%d\n", s==4 ? srcShape[0] : 0, srcShape[s-3], srcShape[s-2], srcShape[s-1] );
   Shape srcStrides = strides;
   int stride = 1;
   for(int i=s-1;i>=0;i--)
   {
      srcStrides[i] = stride;
      stride *= srcShape[i];
   }

   switch(type)
   {
      case Float32: TReorder(((float *)src)         , d, srcShape, srcStrides, order ); break;
      case Float64: TReorder(((double *)src)        , d, srcShape, srcStrides, order ); break;
      case UInt8:   TReorder(((unsigned char *)src) , d, srcShape, srcStrides, order ); break;
      case UInt16:  TReorder(((unsigned short *)src), d, srcShape, srcStrides, order ); break;
      case UInt32:  TReorder(((unsigned int *)src)  , d, srcShape, srcStrides, order ); break;
      case UInt64:  TReorder(((TUInt64 *)src)       , d, srcShape, srcStrides, order ); break;
      case Int8:    TReorder(((signed char *)src)   , d, srcShape, srcStrides, order ); break;
      case Int16:   TReorder(((short *)src)         , d, srcShape, srcStrides, order ); break;
      case Int32:   TReorder(((int *)src)           , d, srcShape, srcStrides, order ); break;
      case Int64:   TReorder(((TInt64 *)src)        , d, srcShape, srcStrides, order ); break;
   }

}




template<typename SRC, typename T>
void TTVisitTensor(const SRC *inSrc,int n, T &visitor)
{
   for(int i=0;i<n;i++)
      visitor.visit(inSrc[i]);
}

template<typename T>
void TVisitTensor(Tensor *tensor, T &visitor)
{
   int n = tensor->elementCount;
   const void *d = tensor->cpuRead();

   switch(tensor->type)
   {
      case Float32: TTVisitTensor(((float *)d)         , n, visitor ); break;
      case Float64: TTVisitTensor(((double *)d)        , n, visitor ); break;
      case UInt8:   TTVisitTensor(((unsigned char *)d) , n, visitor ); break;
      case UInt16:  TTVisitTensor(((unsigned short *)d), n, visitor ); break;
      case UInt32:  TTVisitTensor(((unsigned int *)d)  , n, visitor ); break;
      case UInt64:  TTVisitTensor(((TUInt64 *)d)       , n, visitor ); break;
      case Int8:    TTVisitTensor(((signed char *)d)   , n, visitor ); break;
      case Int16:   TTVisitTensor(((short *)d)         , n, visitor ); break;
      case Int32:   TTVisitTensor(((int *)d)           , n, visitor ); break;
      case Int64:   TTVisitTensor(((TInt64 *)d)        , n, visitor ); break;
   }

}


struct MinVisitor
{
   double result;
   MinVisitor(double init) : result(init) { }

   template<typename T>
   void visit(const T &val)
   {
      if (val<result)
         result = val;
   }
};

double Tensor::getMin()
{
   MinVisitor visitor( getFloatAt(0) );
   TVisitTensor(this, visitor);
   return visitor.result;
}


struct MaxVisitor
{
   double result;
   MaxVisitor(double init) : result(init) { }

   template<typename T>
   void visit(const T &val)
   {
      if (val>result)
         result = val;
   }
};
double Tensor::getMax()
{
   MaxVisitor visitor( getFloatAt(0) );
   TVisitTensor(this, visitor);
   return visitor.result;
}


Tensor *Tensor::cropAndScale(int inWidth, int inHeight, Tensor *inBuffer)
{
   if (shape.size()!=3)
      TensorThrow("cropAndScale - only images of dim 3 are supported");

   Tensor *buffer = makeBuffer(inBuffer, inWidth, inHeight, shape[2], type);

   if (type!=Float32)
      TensorThrow("cropAndScale - only Float32 supported");

   const float *srcImage = (const float *)cpuRead();
   float *destImage = (float *)buffer->cpuWrite();
   int sh = shape[0];
   int sw = shape[1];
   int channels = shape[2];

   for(int y=0;y<inHeight;y++)
   {
      int sy = y*sh/inHeight;
      const float *srcRow = srcImage + sy*strides[0];
      float *dest = destImage + y*buffer->strides[0];

      for(int x=0;x<inWidth;x++)
      {
         int sx = x*sw/inWidth;
         const float *s = srcRow + sx*channels;
         for(int c=0;c<channels;c++)
            *dest++ = s[c];
      }
   }


   return buffer;
}


} // end namespace numerix

