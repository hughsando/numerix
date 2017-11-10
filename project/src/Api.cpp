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

extern "C" void InitIDs()
{
   kind_share(&dataKind,"data");
   kind_share(&tensorKind,"Tensor");

   _id_name = val_id("name");
   _id_type = val_id("type");

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
   Tensor *tensor = new Tensor(0,inType,Shape());

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


