#define IMPLEMENT_API
#include <hx/CffiPrime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <Tensor.h>

#undef max
#undef min

vkind dataKind;
vkind tensorKind;
vkind layerKind;



static int _id_name;
static int _id_type;
static int _id_hx_storeType;
static int _id_hx_elementSize;
static int _id_hx_pointer;
static int _id_length;
static int _id_resultBuffer;

extern "C" void InitIDs()
{
   kind_share(&dataKind,"data");
   kind_share(&tensorKind,"Tensor");
   kind_share(&layerKind,"Layer");

   _id_name = val_id("name");
   _id_type = val_id("type");
   _id_length = val_id("length");
   _id_hx_storeType = val_id("_hx_storeType");
   _id_hx_elementSize = val_id("_hx_elementSize");
   _id_hx_pointer = val_id("_hx_pointer");
   _id_resultBuffer = val_id("resultBuffer");

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


void TensorThrow(const char *err)
{
   val_throw( alloc_string(err) );
}

void fromValue( std::vector<int> &out, value inValue)
{
   int len = (int)val_field_numeric( inValue, _id_length );
   out.resize(len);
   int *ptr= val_array_int(inValue);
   if (ptr)
      for(int i=0;i<len;i++)
         out[i] = ptr[i];
   else
      for(int i=0;i<len;i++)
         out[i] = val_int(val_array_i(inValue,i) );
}


void ClearGc(value inValue)
{
   val_gc(inValue,0);
}

#define TO_TENSOR \
   if (val_kind(inTensor)!=tensorKind) val_throw(alloc_string("object not a tensor")); \
   Tensor *tensor = (Tensor *)val_data(inTensor);

#define TO_TENSOR_NAME(from,to) \
   Tensor *to = (!val_is_null(from) && val_kind(from)==tensorKind) ? (Tensor *)val_data(from) : 0;


void destroyTensor(value inTensor)
{
   TO_TENSOR

   if (tensor)
      tensor->decRef();
}


void tdRelease(value inTensor)
{
   TO_TENSOR_NAME(inTensor, tensor);
   if (tensor)
   {
      tensor->decRef();
      val_gc(inTensor,0);
   }
}

DEFINE_PRIME1v(tdRelease)

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

static int combineTypes(int t0, int t1)
{
   if (t0==t1)
      return t0;
   if (t0<0)
      return t1;
   if (t1<0)
      return t0;

   if ( (t0|t1) & Floating )
   {
      if (t0==Float64 || t1==Float64 || t0==Int64 || t1==Int64)
         return Float64;
      return Float32;
   }
   int size = std::max(t0 & BitsMask, t1 & BitsMask) << 3;
   if ( (t0|t1) & SignedInteger )
      return SignedInteger | size;

   return UnsignedInteger | size;
}


static void findArrayInfo(value item, int *outDims, int &ioType)
{
   ArrayStore storeType = (ArrayStore)(int)val_field_numeric( item, _id_hx_storeType );
   int length = (int)val_field_numeric( item, _id_length );

   if (outDims[0]<length)
      outDims[0] = length;

   if (storeType==arrayObject)
   {
      for(int i=0;i<length;i++)
         findArrayInfo( val_array_i(item,i), outDims+1, ioType );
   }
   else if (storeType==arrayInt)
   {
      int size = (int)val_field_numeric( item, _id_hx_elementSize );
      int type = size==1 ? UInt8 : (SignedInteger | (size<<3));
      ioType = combineTypes(ioType, type);
   }
   else if (storeType==arrayFloat)
   {
      int size = (int)val_field_numeric( item, _id_hx_elementSize );
      int type = size==4 ? Float32 : Float64;
      ioType = combineTypes(ioType, type);
   }
   else
   {
      throw "Could not find numeric array";
   }
}

void fillTensorRow(Tensor *tensor, int inOffset, int haveType, unsigned char *pointer, int length)
{
   int lastSize = tensor->shape[ tensor->shape.size()-1 ];
   if (length>lastSize)
      length = lastSize;
   tensor->fill(haveType, pointer, inOffset, length);
   if (lastSize>length)
      tensor->zero( inOffset + length, lastSize-length );
}

int getArrayType(value inFrom)
{
   ArrayStore storeType = (ArrayStore)(int)val_field_numeric( inFrom, _id_hx_storeType );

   if (storeType==arrayInt)
   {
      int size = (int)val_field_numeric( inFrom, _id_hx_elementSize );
      return size==1 ? UInt8 : (SignedInteger | (size<<3));
   }
   else if (storeType==arrayFloat)
   {
      int size = (int)val_field_numeric( inFrom, _id_hx_elementSize );
      return size==4 ? Float32 : Float64;
   }

   val_throw( alloc_string("non-value data in array") );
   return 0;
}

static void fillTensorDim(Tensor *tensor, value inFrom, int inDim, int inOffset)
{
   if (arrayObject != (ArrayStore)(int)val_field_numeric( inFrom, _id_hx_storeType ))
   {
      val_throw( alloc_string("non-array data in tensor") );
   }
   int length = (int)val_field_numeric( inFrom, _id_length );
   int stride = tensor->strides[inDim];
   int end = std::min(length, tensor->shape[inDim]);

   int nextDim = inDim + 1;
   if (nextDim+1>=tensor->shape.size())
   {
      for(int i=0;i<end;i++)
      {
         value from = val_array_i(inFrom, i);
         int haveType = getArrayType(from);
         int l = (int)val_field_numeric( from, _id_length );
         unsigned char *pointer = 0;
         value ptrVal = val_field(from,_id_hx_pointer);
         if (!val_is_null(ptrVal))
            pointer = (unsigned char *)val_data(ptrVal);
         if (!pointer)
             val_throw( alloc_string("non-pointer data in tensor") );
         int len = (int)val_field_numeric( from, _id_length );
         fillTensorRow(tensor, inOffset, haveType, pointer, len );
         inOffset += stride;
      }
   }
   else
   {
      for(int i=0;i<end;i++)
      {
         value from = val_array_i(inFrom, i);
         fillTensorDim(tensor, from, nextDim, inOffset);
         inOffset += stride;
      }
   }
   if (end<tensor->shape[inDim])
      tensor->zero( inOffset, (tensor->shape[inDim]-end)*stride );
}

value tdFromDynamic(value inFrom, int inType, value inShape)
{

   Shape shape = shapeFromVal(inShape);

   ArrayStore storeType = arrayNull;
   int elementSize = 0;
   int length = 0;
   unsigned char *pointer = 0;

   Shape haveShape;
   int haveType = -1;


   if (!val_is_null(inFrom))
   {
      storeType = (ArrayStore)(int)val_field_numeric( inFrom, _id_hx_storeType );
      elementSize = (int)val_field_numeric( inFrom, _id_hx_elementSize );
      length = (int)val_field_numeric( inFrom, _id_length );

      if (length && elementSize )
      {
         value ptrVal = val_field(inFrom,_id_hx_pointer);
         if (!val_is_null(ptrVal))
            pointer = (unsigned char *)val_data(ptrVal);
      }
   }

   if (storeType==arrayObject && elementSize>0)
   {
      int   dims[64] = { 0 };

      dims[0] = length;
      for(int i=0;i<length;i++)
         findArrayInfo(val_array_i(inFrom,i), dims+1, haveType);

      int idx = 0;
      for(int idx=0; dims[idx]>0; idx++)
         haveShape.push_back(dims[idx]);
   }
   else if (storeType==arrayInt && elementSize>0)
   {
      haveType = SignedInteger | (elementSize<<3);
      haveShape.push_back(length);
   }
   else if (storeType==arrayFloat && elementSize>0)
   {
      haveType = Floating | (elementSize<<3);
      haveShape.push_back(length);
   }
   else if (val_is_int(inFrom))
   {
      haveType = Int32;
      haveShape.push_back(1);
   }
   else if (val_is_float(inFrom))
   {
      haveType = Float64;
      haveShape.push_back(1);
   }
   else if (val_is_null(inFrom))
   {
      // Ok if type provided
   }

   if (inType<0)
       inType = haveType;

   int nType = inType & NumberMask;
   if (nType!=SignedInteger && nType!=UnsignedInteger && nType!=Floating)
      val_throw( alloc_string("Could not determine tensor type"));

   if (shape.size()==0)
      shape = haveShape;

   if (haveShape.size()>shape.size())
      val_throw( alloc_string("too much data for provided shape"));

   Tensor *tensor = new Tensor(inType,shape);
   value result = alloc_abstract(tensorKind,tensor);
   val_gc(result, destroyTensor);

   if (storeType==arrayNull)
   {
      if (haveType & (UnsignedInteger|SignedInteger))
         tensor->setInt32(val_int(inFrom),0, tensor->elementCount);
      else
         tensor->setFloat64(val_float(inFrom), 0, tensor->elementCount);
   }
   else if (shape.size()==1 && haveShape.size()==1 && pointer)
   {
      fillTensorRow(tensor, 0, haveType, pointer, length);
   }
   else
   {
      if (haveShape.size()!=shape.size())
         val_throw( alloc_string("provided data does not match the requested shape"));
      /*
      if (haveShape.size()<shape.size())
      {
         broadcastFill(tensor, inFrom, haveShape, 0);
      }
      */

      fillTensorDim(tensor, inFrom, 0, 0);
   }

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
DEFINE_PRIME1(tdGetType)


value tdGetData(value inTensor)
{
   TO_TENSOR
   return alloc_abstract(dataKind,tensor->data);
}
DEFINE_PRIME1(tdGetData)


void tdFillData(value inTensor, value outBuffer)
{
   TO_TENSOR

   CffiBytes bytes = getByteData(outBuffer);

   if (!bytes.data)
      TensorThrow("tdFillData - bad buffer");
   int len = tensor->elementCount * tensor->elementSize;
   if (len<bytes.length)
      len = bytes.length;
   memcpy(bytes.data, tensor->data, len);
}
DEFINE_PRIME2v(tdFillData)



void tdPrint(value inTensor, int maxElems)
{
   TO_TENSOR
   return tensor->print(maxElems);
}
DEFINE_PRIME2v(tdPrint);


value tdReorder(value inTensor, value inNewOrder)
{
   TO_TENSOR
   std::vector<int> newOrder;
   fromValue(newOrder, inNewOrder);
   Tensor *result =  tensor->reorder(newOrder);
   return alloc_abstract(tensorKind,result);
}
DEFINE_PRIME2v(tdReorder);

// ----------- Layer


#define TO_LAYER \
   if (val_kind(inLayer)!=layerKind) val_throw(alloc_string("object not a layer")); \
   Layer *layer = (Layer *)val_data(inLayer);

void destroyLayer(value inLayer)
{
   TO_LAYER

   if (layer)
      delete layer;
}


value layCreateConv2D(value inStrides, int activation, int padding, value inWeights, value inBias)
{
   TO_TENSOR_NAME(inWeights, weights);
   if (!weights)
      TensorThrow("Conv2D - invalid weights");

   TO_TENSOR_NAME(inBias, bias);

   int sx = 1;
   int sy = 1;
   if (val_is_int(inStrides))
      sx = sy = val_int(inStrides);
   else if (!val_is_null(inStrides))
   {
      Shape strides;
      fromValue(strides, inStrides);
      sy = strides.size() > 0 ? strides[0] : 1;
      sx = strides.size() > 1 ? strides[1] : sy;
   }

   Layer *layer = Layer::createConv2D(sx, sy, (Activation)activation, (Padding)padding, weights, bias);

   value result = alloc_abstract(layerKind, layer);
   val_gc(result, destroyLayer);
   return result;
}
DEFINE_PRIME5(layCreateConv2D);

value layRun(value inLayer, value inOwner, value inSrc0)
{
   TO_LAYER
   TO_TENSOR_NAME(inSrc0, src0)

   value valBuf = val_field(inOwner, _id_resultBuffer);
   TO_TENSOR_NAME(valBuf, buffer);

   Tensor *result = layer->run(src0, buffer);
   if (result!=buffer)
   {
      if (buffer)
         tdRelease(valBuf);
      if (result)
      {
         valBuf = alloc_abstract(tensorKind, result);
         val_gc(valBuf, destroyTensor);
      }
      else
         valBuf = alloc_null();

      alloc_field(inOwner, _id_resultBuffer, valBuf);
   }

   return valBuf;
}
DEFINE_PRIME3(layRun);

void layRelease(value inLayer)
{
   if (!val_is_null(inLayer))
   {
      TO_LAYER

      delete layer;
      ClearGc(inLayer);
   }
}
DEFINE_PRIME1v(layRelease);



