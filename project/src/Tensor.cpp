#include <Tensor.h>



Tensor::Tensor(u8 *inData, int inType, const Shape &inShape )
   : data(inData), type(inType), shape(inShape)
{
   refCount = 1;
   elementCount = 1;
   if (shape.size()==0)
      shape.push_back(1);
   else
      for(int i=0;i<shape.size();i++)
         elementCount *= shape[i];
}

Tensor::~Tensor()
{
   // TODO aligned?
   if (data)
      free(data);
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


