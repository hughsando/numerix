#define IMPLEMENT_API
#include <hx/CffiPrime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <Tensor.h>

vkind dataKind;
vkind tensorKind;



static int _id_name;
static int _id_type;
static int _id_hx_storeType;
static int _id_hx_elementSize;
static int _id_hx_pointer;
static int _id_length;

extern "C" void InitIDs()
{
   kind_share(&dataKind,"data");
   kind_share(&tensorKind,"Tensor");

   _id_name = val_id("name");
   _id_type = val_id("type");
   _id_length = val_id("length");
   _id_hx_storeType = val_id("_hx_storeType");
   _id_hx_elementSize = val_id("_hx_elementSize");
   _id_hx_pointer = val_id("_hx_pointer");

}

Shape shapeFromVal(value inShape)
{
   int *data = val_array_int(inShape);
   if (!data)
      return Shape();

   int l = val_array_size(inShape);
   if (l<0 || l>64)
      val_throw( alloc_string("Invalid shape length"));
   Shape result(l);
   for(int i=0;i<l;i++)
   {
      result[i] = data[i];
      if (result[i]<1)
          val_throw( alloc_string("Invalid shape dimension"));
   }

   return result;
}

DEFINE_ENTRY_POINT(InitIDs)


extern "C" int numerix_register_prims()
{
   InitIDs();
   return 0;
}


void ClearGc(value inValue)
{
   val_gc(inValue,0);
}

#define TO_TENSOR \
   if (val_kind(inTensor)!=tensorKind) val_throw(alloc_string("object not a tensor")); \
   Tensor *tensor = (Tensor *)val_data(inTensor);

void destroyTensor(value inTensor)
{
   TO_TENSOR

   if (tensor)
      tensor->decRef();
}


void tdRelease(value inTensor)
{
   TO_TENSOR

   tensor->decRef();
   val_gc(inTensor,0);
}

DEFINE_PRIME1v(tdRelease)

value tdFromDynamic(value inFrom, int inType, value inShape)
{
   enum ArrayStore
   {
      arrayNull = 0,
      arrayEmpty,
      arrayFixed,
      arrayBool,
      arrayInt,
      arrayFloat,
      arrayString,
      arrayObject,
   };

   Shape shape = shapeFromVal(inShape);

   ArrayStore storeType = (ArrayStore)(int)val_field_numeric( inFrom, _id_hx_storeType );
   int elementSize = (int)val_field_numeric( inFrom, _id_hx_elementSize );
   int length = (int)val_field_numeric( inFrom, _id_length );
   int haveType = -1;
   unsigned char *pointer = 0;

   if (length && elementSize)
   {
      value ptrVal = val_field(inFrom,_id_hx_pointer);
      if (!val_is_null(ptrVal))
         pointer = (unsigned char *)val_data(ptrVal);
   }

   if (storeType==arrayInt && elementSize>0)
   {
      haveType = SignedInteger | (elementSize<<3);
      if (shape.size()==0)
         shape.push_back(length);
   }
   else if (storeType==arrayFloat && elementSize>0)
   {
      haveType = Floating | (elementSize<<3);
      if (shape.size()==0)
         shape.push_back(length);
   }
   if (inType<0)
       inType = haveType;

   int nType = inType & NumberMask;
   if (nType!=SignedInteger && nType!=UnsignedInteger && nType!=Floating)
      val_throw( alloc_string("Could not determine tensor type"));

   Tensor *tensor = new Tensor(inType,shape);
   if (haveType>=0 && pointer)
   {
      tensor->fill(haveType, pointer, length);
   }

   value result = alloc_abstract(tensorKind,tensor);
   val_gc(result, destroyTensor);
   return result;
}
DEFINE_PRIME3(tdFromDynamic)

int tdGetDimCount(value inTensor)
{
   TO_TENSOR
   return tensor->shape.size();
}
DEFINE_PRIME1(tdGetDimCount);

int tdGetDimAt(value inTensor,int inIndex)
{
   TO_TENSOR
   return tensor->shape[inIndex];
}
DEFINE_PRIME2(tdGetDimAt);

int tdGetElementCount(value inTensor)
{
   TO_TENSOR
   return tensor->elementCount;
}
DEFINE_PRIME1(tdGetElementCount);

int tdGetType(value inTensor)
{
   TO_TENSOR
   return tensor->type;
}
DEFINE_PRIME1(tdGetType);


