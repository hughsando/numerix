#define IMPLEMENT_API
#include <hx/CFFIPrime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <Tensor.h>
#include <Layer.h>

#ifdef NX_MOVIDIUS
#include "mvnc.h"
#endif

#undef max
#undef min

namespace numerix
{

vkind dataKind;
vkind tensorKind;
vkind layerKind;

bool enableGpu = true;


static int _id_name;
static int _id_type;
static int _id_hx_storeType;
static int _id_hx_elementSize;
static int _id_hx_pointer;
static int _id_length;
static int _id_resultBuffer;
static int _id_x;
static int _id_y;
static int _id_w;
static int _id_h;
static int _id_prob;
static int _id_classId;

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
   _id_x = val_id("x");
   _id_y = val_id("y");
   _id_w = val_id("w");
   _id_h = val_id("h");
   _id_classId = val_id("classId");
   _id_prob = val_id("prob");
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


value allocTensor(Tensor *inTensor)
{
   value result = alloc_abstract(tensorKind,inTensor);
   val_gc(result, destroyTensor);
   return result;
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

void fillTensorData(Tensor *tensor, int haveType, unsigned char *pointer, int length)
{
   int count = tensor->elementCount;
   if (length>count)
      length = count;
   tensor->fill(haveType, pointer, 0, length);
   if (count>length)
      tensor->zero( length, count-length );
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


   CffiBytes bytes;

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
      if (!pointer)
         bytes = getByteData(inFrom);
   }


   if (storeType==arrayObject && elementSize>0 && !bytes.data)
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
   value result = allocTensor(tensor);

   if (storeType==arrayNull && !bytes.data)
   {
      if (haveType & (UnsignedInteger|SignedInteger))
         tensor->setInt32(val_int(inFrom),0, tensor->elementCount);
      else
         tensor->setFloat64(val_float(inFrom), 0, tensor->elementCount);
   }
   else if (haveShape.size()==1 && pointer)
   {
      fillTensorData(tensor, haveType, pointer, length);
   }
   else if (bytes.data)
   {
      int elems = bytes.length/tensor->elementSize;
      if (elems!=tensor->elementCount)
         TensorThrow("provided data length does not match the requested shape");

      tensor->fill(inType, bytes.data, 0, tensor->elementCount);
   }
   else
   {
      if (haveShape.size()!=shape.size())
         TensorThrow("provided data does not match the requested shape");
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
   if (inIndex>=tensor->shape.size())
      return -1;
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
   // TODO - usage read/write
   return alloc_abstract(dataKind,tensor->cpuWritePart());
}
DEFINE_PRIME1(tdGetData)


double tdGetMin(value inTensor)
{
   TO_TENSOR
   return tensor->getMin();
}
DEFINE_PRIME1(tdGetMin)

double tdGetMax(value inTensor)
{
   TO_TENSOR
   return tensor->getMax();
}
DEFINE_PRIME1(tdGetMax)

double tdAt(value inTensor, int index)
{
   TO_TENSOR
   return tensor->getFloatAt(index);
}
DEFINE_PRIME2(tdAt)

void tdSetAt(value inTensor, int index, double inVal,int count)
{
   TO_TENSOR
   tensor->setFloat64(inVal,index,count);
}
DEFINE_PRIME4v(tdSetAt)



void tdSetFlat(value inTensor)
{
   TO_TENSOR
   tensor->setFlat();
}
DEFINE_PRIME1v(tdSetFlat)

void tdSetShape(value inTensor,value inShape)
{
   TO_TENSOR

   Shape shape;
   fromValue(shape, inShape);

   tensor->setShape(shape);
}
DEFINE_PRIME2v(tdSetShape)


value tdResizeAxis(value inTensor,int inAxis, int inSize, int inDataPos)
{
   TO_TENSOR

   return allocTensor( tensor->resizeAxis(inAxis, inSize, inDataPos) );
}
DEFINE_PRIME4(tdResizeAxis)




void tdFillData(value inTensor, value outBuffer)
{
   TO_TENSOR

   CffiBytes bytes = getByteData(outBuffer);

   if (!bytes.data)
      TensorThrow("tdFillData - bad buffer");
   int len = tensor->elementCount * tensor->elementSize;
   if (len<bytes.length)
   {
      len = bytes.length;
      memcpy(bytes.data, tensor->cpuWritePart(), bytes.length);
   }
   else
   {
      memcpy(bytes.data, tensor->cpuWrite(), len);
   }
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
   return allocTensor(result);
}
DEFINE_PRIME2v(tdReorder);


value tdCropAndScale(value inTensor, int inWidth, int inHeight, value inBuffer)
{
   TO_TENSOR
   TO_TENSOR_NAME(inBuffer,buffer)

   Tensor *result =  tensor->cropAndScale(inWidth, inHeight, buffer);

   if (result==buffer)
      return inBuffer;

   return allocTensor(result);
}
DEFINE_PRIME4(tdCropAndScale);


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

value allocLayer(Layer *inLayer)
{
   value result = alloc_abstract(layerKind, inLayer);
   val_gc(result, destroyLayer);
   return result;
}

value layCreateConv2D(value inStrides, int inActivation, int inPadding, value inWeights, value inPWeights, value inBias)
{
   TO_TENSOR_NAME(inWeights, weights);
   if (!weights)
      TensorThrow("Conv2D - invalid weights");

   TO_TENSOR_NAME(inBias, bias);
   TO_TENSOR_NAME(inPWeights, pweights);

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

   Layer *layer = 0;
   Activation activation = (Activation)inActivation;
   Padding padding = (Padding)inPadding;
   #ifdef NX_GPU
   if (enableGpu && !pweights && gpuInit())
      layer = gpuCreateConv2D(sx,sy, activation, padding, weights, bias);
   else
   #endif
   layer = Layer::createConv2D(sx, sy, activation, padding, weights, pweights, bias);

   return allocLayer(layer);
}
DEFINE_PRIME6(layCreateConv2D);



void layConv2DSetNorm(value inLayer, value inScales, value inMeans, value inVars)
{
   TO_LAYER
   TO_TENSOR_NAME(inScales, scales);
   TO_TENSOR_NAME(inMeans, means);
   TO_TENSOR_NAME(inVars, vars);

   if (!scales || !means || !vars)
      TensorThrow("Conv2D - invalid normalization");

   layer->setNormalization(scales, means, vars);
}
DEFINE_PRIME4v(layConv2DSetNorm);



value layCreateMaxPool(value inSize, value inStrides, int inPadding)
{
   int sx = 1;
   int sy = 1;
   if (val_is_int(inSize))
      sx = sy = val_int(inSize);
   else if (!val_is_null(inSize))
   {
      Shape size;
      fromValue(size, inSize);
      sy = size.size() > 0 ? size[0] : 1;
      sx = size.size() > 1 ? size[1] : sy;
   }

   int stepX = sx;
   int stepY = sy;
   if (val_is_int(inStrides))
      stepX = stepY = val_int(inStrides);
   else if (!val_is_null(inStrides))
   {
      Shape strides;
      fromValue(strides, inStrides);
      stepY = strides.size() > 0 ? strides[0] : 1;
      stepX = strides.size() > 1 ? strides[1] : sy;
   }

   Layer *layer = 0;
   Padding padding = (Padding)inPadding;
   #ifdef NX_GPU
   if (enableGpu && gpuInit())
      layer = gpuCreateMaxPool(sx, sy, stepX, stepY, padding);
   else
   #endif
       layer = Layer::createMaxPool(sx, sy, stepX, stepY, padding);

   return allocLayer(layer);
}
DEFINE_PRIME3(layCreateMaxPool);



value layCreateConcat()
{
   Layer *layer = 0;
   #ifdef NX_GPU
   if ( enableGpu && gpuInit())
      layer = gpuCreateConcat();
   else
   #endif
      layer = Layer::createConcat();

   return allocLayer(layer);
}
DEFINE_PRIME0(layCreateConcat);



value layCreateReorg(int inStride)
{
   Layer *layer = Layer::createReorg(inStride);

   return allocLayer(layer);
}
DEFINE_PRIME1(layCreateReorg);



value layCreateYolo(value inAnchors, int inNum, int inClassCount)
{
   std::vector<float> anchors(val_array_size(inAnchors));
   for(int i=0;i<anchors.size();i++)
      anchors[i] = val_number( val_array_i(inAnchors,i) );

   Layer *layer = 0;
   #ifdef NX_GPU
   if ( enableGpu && gpuInit())
      layer = gpuCreateYolo(anchors, inNum, inClassCount,0.25);
   else
   #endif
      layer = Layer::createYolo(anchors, inNum, inClassCount,0.25);

   return allocLayer(layer);
}
DEFINE_PRIME3(layCreateYolo);


value layGetBoxes(value inLayer)
{
   TO_LAYER;

   Boxes boxes;
   layer->getBoxes(boxes);

   value result = alloc_array( boxes.size() );
   for(int i=0;i<boxes.size();i++)
   {
      BBox &b = boxes[i];
      value v = alloc_empty_object();
      alloc_field(v, _id_x, alloc_float(b.x) );
      alloc_field(v, _id_y, alloc_float(b.y) );
      alloc_field(v, _id_w, alloc_float(b.w) );
      alloc_field(v, _id_h, alloc_float(b.h) );
      alloc_field(v, _id_prob, alloc_float(b.prob) );
      alloc_field(v, _id_classId, alloc_int(b.classId) );
      val_array_set_i(result, i, v);
   }

   return result;
}
DEFINE_PRIME1(layGetBoxes);

void laySetPadInput(value inLayer)
{
   TO_LAYER
   layer->setPadInput();
}
DEFINE_PRIME1v(laySetPadInput);



value layRun(value inLayer, value inOwner, value inSrc)
{
   TO_LAYER
   Tensor *src0 = 0;
   Tensor *src1 = 0;

   if (val_is_array(inSrc))
   {
      TO_TENSOR_NAME(val_array_i(inSrc,0), s0);
      src0 = s0;
      TO_TENSOR_NAME(val_array_i(inSrc,1), s1);
      src1 = s1;
   }
   else
   {
      TO_TENSOR_NAME(inSrc, src);
      src0 = src;
   }

   value valBuf = val_field(inOwner, _id_resultBuffer);
   TO_TENSOR_NAME(valBuf, buffer);

   Tensor *result = src1==0 ?
                       layer->run(src0, buffer) :
                       layer->run(src0, src1, buffer);

   if (result!=buffer)
   {
      if (buffer)
         tdRelease(valBuf);

      if (result)
         valBuf = allocTensor(result);
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

// Movidius device ...

HxString moviGetDeviceName(int inIndex)
{
   #ifndef NX_MOVIDIUS
   TensorThrow("Device not supported on this platform");
   #else
   #define NAME_SIZE 100

   void *deviceHandle;
   char devName[NAME_SIZE];
   mvncStatus retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
   if (retCode != MVNC_OK)
   {
      char buf[1024];
      sprintf(buf, "Error - No NCS devices found :%d.", retCode);
      TensorThrow(buf);
   }

   return devName;
   #endif
}
DEFINE_PRIME1(moviGetDeviceName);

value layCreateMovidius(HxString devName, value inGraphDef, value inOutputShape)
{
   #ifndef NX_MOVIDIUS
   TensorThrow("Device not supported on this platform");
   #else

   CffiBytes bytes = getByteData(inGraphDef);
   if (!bytes.data)
      TensorThrow("layCreateMovidius - bad buffer");

   Shape shape;
   fromValue(shape, inOutputShape);

   return allocLayer( Layer::createMovidius(devName.c_str(), bytes.data, bytes.length, shape) );
   #endif
}
DEFINE_PRIME3(layCreateMovidius);

} // end namespace numerix
