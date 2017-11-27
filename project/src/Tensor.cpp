#include <Tensor.h>
#include <hx/Thread.h>

#include <stdexcept>

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
      CShape &s = inBuffer->shape;
      if (s[0]==inW && s[1]==inH && s[2]==inChannels)
         return inBuffer;
   }

   Shape s(3);
   s[0] = inW;
   s[1] = inH;
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
   void *d = getCpu();
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
   void *d = getCpu();
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
   u8 *d = getCpu();

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
   memset( getCpu() + inOffsetElem*elementSize, 0, inCount*elementSize );
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
      void *d = getCpu();
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
      void *d = getCpu();
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
   void *d = t->getCpu();
   void *src = getCpu();
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
   void *d = tensor->getCpu();

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
   Tensor *buffer = makeBuffer(inBuffer, inWidth, inHeight, shape[2], type);

   printf("TODO - cropAndScale %d %d %d\n", inWidth, inHeight, shape[2]);

   return buffer;
}


} // end namespace numerix

