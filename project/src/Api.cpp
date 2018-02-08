#define IMPLEMENT_API
#include <hx/CFFIPrime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <Tensor.h>
#include <Layer.h>

#include <OCL.h>

#undef min
#undef max
#undef STRICT

#ifdef NX_CAFFE
#include "caffe/caffe.pb.h"
#include <fcntl.h>
#include <io.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#endif

#ifdef NX_MOVIDIUS
#include "mvnc.h"
#endif



namespace numerix
{

vkind dataKind;
vkind tensorKind;
vkind layerKind;
vkind oclDeviceKind;
vkind oclPlatformKind;
vkind oclContextKind;

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
static int _id_id;

extern "C" void InitIDs()
{
   kind_share(&dataKind,"data");
   kind_share(&tensorKind,"Tensor");
   kind_share(&layerKind,"Layer");
   kind_share(&oclDeviceKind,"oclDevice");
   kind_share(&oclPlatformKind,"oclPlatform");
   kind_share(&oclContextKind,"oclContext");

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
   _id_id = val_id("id");
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

double tdGetAt(value inTensor, int idx0, value idx1, value idx2, value idx3, value idx4)
{
   TO_TENSOR
   if (val_is_null(idx1))
      return tensor->getFloatAt(idx0);
   if (val_is_null(idx2))
      return tensor->getFloat(idx0,val_int(idx1));
   if (val_is_null(idx3))
      return tensor->getFloat(idx0,val_int(idx1),val_int(idx2));
   if (val_is_null(idx4))
      return tensor->getFloat(idx0,val_int(idx1),val_int(idx2),val_int(idx3));
   return tensor->getFloat(idx0,val_int(idx1),val_int(idx2),val_int(idx3),val_int(idx4));
}
DEFINE_PRIME6(tdGetAt)


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


void nxEnableGpu(bool inEnable)
{
   enableGpu = inEnable;
}
DEFINE_PRIME1v(nxEnableGpu);


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

value layCreateConv2D(value inStrides, int inActivation, int inPadding, value inWeights, value inPWeights, value inBias, bool inAllowTransform)
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

   #ifdef NX_OPENCL
   if (!layer && !pweights && OclContext::hasCurrent())
      layer = oclCreateConv2D(sx,sy, activation, padding, weights, bias);
   #endif

   #ifdef NX_GPU
   if (!layer && enableGpu && !pweights && gpuInit())
      layer = gpuCreateConv2D(sx,sy, activation, padding, weights, bias);
   #endif

   if (!layer)
      layer = Layer::createConv2D(sx, sy, activation, padding, weights, pweights, bias, inAllowTransform);

   return allocLayer(layer);
}
DEFINE_PRIME7(layCreateConv2D);



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

void layConv2DSetActication(value inLayer, int inActivation)
{
   TO_LAYER
   layer->setActivation((Activation)inActivation);
}
DEFINE_PRIME2v(layConv2DSetActication)


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
   #ifdef NX_OPENCL
   if (OclContext::hasCurrent())
      layer = oclCreateMaxPool(sx, sy, stepX, stepY, padding);
   #endif

   #ifdef NX_GPU
   if (!layer && enableGpu && gpuInit())
      layer = gpuCreateMaxPool(sx, sy, stepX, stepY, padding);
   #endif

   if (!layer)
       layer = Layer::createMaxPool(sx, sy, stepX, stepY, padding);

   return allocLayer(layer);
}
DEFINE_PRIME3(layCreateMaxPool);



value layCreateGlobalPool(bool inAverage)
{
   Layer *layer = 0;
   layer = Layer::createGlobalPool();

   return allocLayer(layer);
}
DEFINE_PRIME1(layCreateGlobalPool);


double layGetRunTime(value inLayer)
{
   TO_LAYER
   return layer->getRunTime();
}
DEFINE_PRIME1(layGetRunTime);


value layCreateSoftmax()
{
   Layer *layer = 0;
   layer = Layer::createSoftmax();

   return allocLayer(layer);
}
DEFINE_PRIME0(layCreateSoftmax);



value layCreateConcat()
{
   Layer *layer = 0;
   #ifdef NX_GPU
   if ( enableGpu && !layer && gpuInit())
      layer = gpuCreateConcat();
   #endif
   #ifdef NX_OPENCL
   if (!layer && OclContext::hasCurrent())
      layer = oclCreateConcat( );
   #endif
   if (!layer)
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

void layAccurateTimes(bool inAccurate)
{
   Layer::accurateTimes = inAccurate;
}
DEFINE_PRIME1v(layAccurateTimes)


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




void layEnablePerLayerTiming(bool inLayer)
{
   Layer::openCLTimingEvents = inLayer;
}
DEFINE_PRIME1v(layEnablePerLayerTiming);



// Movidius device ...

HxString moviGetDeviceName(int inIndex)
{
   #ifndef NX_MOVIDIUS
   TensorThrow("Device not supported on this platform");
   return "";
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
   return alloc_null();
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


value oclGetPlatforms()
{
   #ifdef NX_OPENCL
   OclPlatformList platforms = oclGetPlatformList();

   int n = platforms.size();
   value result = alloc_array(n);
   for(int p=0;p<n;p++)
   {
      value platform = alloc_empty_object();

      alloc_field( platform, _id_id, alloc_abstract(oclPlatformKind, platforms[p] ) );
      OclProps props;
      oclGetPlatformProps(platforms[p], props);
      for(int i=0;i<props.size();i++)
         alloc_field( platform, val_id( props[i].key.c_str() ), alloc_string( props[i].value.c_str() ) );

      OclDeviceList devices = oclGetPlatformDevices(platforms[p]);
      value deviceArray = alloc_array( devices.size() );
      for(int i=0;i<devices.size();i++)
      {
         value device = alloc_empty_object();

         OclProps props;
         int computeUnits = 0;
         oclGetDeviceProps( devices[i], props, computeUnits);
         for(int i=0;i<props.size();i++)
            alloc_field( device, val_id( props[i].key.c_str() ), alloc_string( props[i].value.c_str() ) );

         alloc_field( device, val_id("computeUnits"), alloc_int(computeUnits) );
         alloc_field( device, _id_id, alloc_abstract(oclDeviceKind, devices[i] ) );

         val_array_set_i(deviceArray,i,device);
      }
      alloc_field(platform, val_id("devices"), deviceArray);

      val_array_set_i(result, p, platform);
   }
   return result;
   #else
   return alloc_array(0);
   #endif
}
DEFINE_PRIME0(oclGetPlatforms);


#define TO_OCL_CONTEXT \
   if (val_kind(inOclContext)!=oclContextKind) val_throw(alloc_string("object not a ocl context")); \
   OclContext *context = (OclContext *)val_data(inOclContext);

#define TO_PLATFORM \
   if (val_kind(inPlatform)!=oclPlatformKind) val_throw(alloc_string("object not a ocl platform")); \
   void *platform = val_data(inPlatform);

#define TO_DEVICES \
   int n = val_array_size(inDevices); \
   OclDeviceList devices(n); \
   for(int i=0;i<n;i++) \
   { \
      value device = val_array_i(inDevices,i); \
      if (val_kind(device)!=oclDeviceKind) \
         TensorThrow("object not a device"); \
      devices[i] = val_data(device); \
   }


void releaseOclContext(value inOclContext)
{
   TO_OCL_CONTEXT

   if (context)
      context->decRef();
}


value oclCreateContext(value inPlatform, value inDevices)
{
   #ifdef NX_OPENCL

   TO_PLATFORM

   TO_DEVICES

   void *ctx = OclContext::create(platform, devices);
   if (!ctx)
      TensorThrow("Could not create OpenCL context");

   value result = alloc_abstract( oclContextKind, ctx );
   val_gc(result, releaseOclContext );
   return result;

   #else
   TensorThrow("Opencl - not supported");
   return alloc_null();
   #endif
}

DEFINE_PRIME2(oclCreateContext);


void oclSetCurrent(value inOclContext)
{
   #ifdef NX_OPENCL
   if (val_is_null(inOclContext))
      OclContext::setCurrent(0);
   else
   {
      TO_OCL_CONTEXT
      OclContext::setCurrent(context);
   }
   #else
   TensorThrow("Opencl - not supported");
   #endif
}

DEFINE_PRIME1v(oclSetCurrent);

template<typename T>
value make_int_array(T &val)
{
   int n = val.size();
   if (n==0)
      return alloc_null();
   value array = alloc_array(n);
   for(int i=0;i<n;i++)
      val_array_set_i(array,i, alloc_int((int)val.Get(i)) );
   return array;
}

template<typename T>
value make_string_array(T &val)
{
   int n = val.size();
   if (n==0)
      return alloc_null();
   value array = alloc_array(n);
   for(int i=0;i<n;i++)
      val_array_set_i(array,i, alloc_string(val.Get(i).c_str()) );
   return array;
}

#ifdef NX_CAFFE


template<typename T>
static bool fillBlobs(T &layer, value array)
{
   bool result = false;
   int blobs = layer.blobs_size();

   for(int b=0;b<blobs;b++)
   {
      auto &blob = layer.blobs(b);

      if (blob.data_size() || blob.double_data_size())
      {
         int dims = blob.shape().dim_size();
         Shape shape(dims);
         for(int d=0;d<dims;d++)
            shape[d] = blob.shape().dim(d);

         int type = blob.data_size() ? Float32 : Float64;
         Tensor *tensor = new Tensor(type, shape);
         value ten = allocTensor(tensor);

         int count = tensor->elementCount;
         if (blob.data_size())
         {
            for(int i=0;i<count;i++)
               tensor->setFloat64(blob.data(i),i,1);
         }
         else
         {
            for(int i=0;i<count;i++)
               tensor->setFloat64(blob.double_data(i),i,1);
         }


         //TODO - setdata
         //printf("Made %d\n", dims);

         val_array_push(array, ten);
         result = true;
      }
   }
   return result;
}


static void loadCaffeNet(caffe::NetParameter net,value m,bool weightsOnly )
{
   int n = net.layer_size();

   if (weightsOnly)
   {
      int srcN = val_array_size(m);
      if (srcN==0)
         return;
      for(int i=0;i<n;i++)
      {
         auto &layer = net.layer(i);
         int blobs = layer.blobs_size();
         if (blobs)
         {
            std::string name = layer.name();
            value item = 0;
            for(int j=0;j<srcN;j++)
            {
               value test = val_array_i(m,j);
               if (name==val_string( val_field(test, _id_name) ) )
               {
                  item = test;
                  break;
               }
            }
            if (item==0)
            {
               //printf("no match %s\n", name.c_str());
               continue;
            }

            value weights = alloc_array(0);
            if (fillBlobs(layer, weights))
               alloc_field(item, val_id("weights"), weights);
         }
      }
      return;
   }


   std::map<std::string,bool> knownTypes;
   knownTypes["ReLU"] = true;
   knownTypes["Data"] = true;
   knownTypes["Accuracy"] = true;
   knownTypes["Dropout"] = true;
   knownTypes["Split"] = true;
   knownTypes["Softmax"] = true;
   knownTypes["Scale"] = true;
   knownTypes["BatchNorm"] = true;

   value tops = make_string_array(net.input());
   if (!val_is_null(tops))
   {
      value lay = alloc_empty_object();
      alloc_field(lay, _id_type, alloc_string("Input") );
      alloc_field(lay, val_id("input_dim"), make_int_array(net.input_dim()));
      alloc_field(lay, val_id("top"), tops );
      val_array_push(m,  lay);
   }

   //printf("Layers %d\n", n);
   for(int i=0;i<n;i++)
   {
      auto &layer = net.layer(i);
      const std::string layer_type = layer.type();

      value lay = alloc_empty_object();
      alloc_field(lay, _id_type, alloc_string( layer_type.c_str() ) );

      std::string name = layer.name();
      alloc_field(lay, _id_name, alloc_string(name.c_str()));

      alloc_field(lay, val_id("top"), make_string_array( layer.top() ) );
      alloc_field(lay, val_id("bottom"), make_string_array( layer.bottom() ) );


      int blobs = layer.blobs_size();
      if (blobs)
      {
         value shapes = alloc_array(blobs);
         bool hasWeights = false;
         value weights = alloc_array(0);
         if (fillBlobs(layer, weights))
            alloc_field(lay, val_id("weights"), weights);
         for(int b=0;b<blobs;b++)
         {
            auto &blob = layer.blobs(b);

            val_array_set_i(shapes, b, make_int_array(blob.shape().dim()));

            if (blob.data_size() || blob.double_data_size())
            {
               int dims = blob.shape().dim_size();
               Shape shape(dims);
               for(int d=0;d<dims;d++)
                  shape[d] = blob.shape().dim(d);

               int type = blob.data_size() ? Float32 : Float64;
               Tensor *tensor = new Tensor(type, shape);
               value ten = allocTensor(tensor);

               //TODO - setdata

               val_array_push(weights, ten);
            }
         }
         alloc_field(lay, val_id("shapes"), shapes);
      }


      if (knownTypes[layer_type])
      {
         // ignore
      }
      else if (layer_type=="Input")
      {
         auto &param = layer.input_param();
         if (param.shape_size()>0)
         {
            auto &shape = param.shape(0);
            alloc_field(lay, val_id("input_dim"),make_int_array(shape.dim()));
         }
      }
      else if (layer_type=="Convolution")
      {
         auto &param = layer.convolution_param();
         alloc_field(lay, val_id("filters"), alloc_int(param.num_output()));
         alloc_field(lay, val_id("pad"), make_int_array(param.pad()));
         alloc_field(lay, val_id("size"), make_int_array(param.kernel_size()));
         alloc_field(lay, val_id("stride"), make_int_array(param.stride()));
         alloc_field(lay, val_id("dilation"), make_int_array(param.dilation()));
         if (param.has_pad_w())
            alloc_field(lay, val_id("pad_w"), alloc_int(param.pad_w()));
         if (param.has_pad_h())
            alloc_field(lay, val_id("pad_h"), alloc_int(param.pad_h()));
         if (param.has_kernel_w())
            alloc_field(lay, val_id("kernel_w"), alloc_int(param.kernel_w()));
         if (param.has_kernel_h())
            alloc_field(lay, val_id("kernel_h"), alloc_int(param.kernel_h()));
         if (param.has_stride_w())
            alloc_field(lay, val_id("stride_w"), alloc_int(param.stride_w()));
         if (param.has_stride_h())
            alloc_field(lay, val_id("stride_h"), alloc_int(param.stride_h()));
         alloc_field(lay, val_id("axis"), alloc_int(param.axis()));
      }
      else if (layer_type=="Pooling")
      {
         auto &param = layer.pooling_param();
         alloc_field(lay, val_id("method"), alloc_int(param.pool()));
         if (param.has_pad())
            alloc_field(lay, val_id("pad"), alloc_int(param.pad()));
         if (param.has_pad_w())
            alloc_field(lay, val_id("pad_w"), alloc_int(param.pad_w()));
         if (param.has_pad_h())
            alloc_field(lay, val_id("pad_h"), alloc_int(param.pad_h()));
         if (param.has_stride())
            alloc_field(lay, val_id("stride"), alloc_int(param.stride()));
         if (param.has_stride_w())
            alloc_field(lay, val_id("stride_w"), alloc_int(param.stride_w()));
         if (param.has_stride_h())
            alloc_field(lay, val_id("stride_h"), alloc_int(param.stride_h()));

         if (param.has_kernel_size())
            alloc_field(lay, val_id("size"), alloc_int(param.kernel_size()));
         if (param.has_kernel_w())
            alloc_field(lay, val_id("kernel_w"), alloc_int(param.kernel_w()));
         if (param.has_kernel_h())
            alloc_field(lay, val_id("kernel_h"), alloc_int(param.kernel_h()));

         if (param.has_global_pooling())
            alloc_field(lay, val_id("global_pooling"), alloc_bool(param.global_pooling()));
      }
      else if (layer_type=="Concat")
      {
         auto &param = layer.concat_param();
         if (param.has_axis())
            alloc_field(lay, val_id("axis"), alloc_int(param.axis()));
      }
      else if (layer_type=="Eltwise")
      {
         auto &param = layer.eltwise_param();
         int op = param.operation();
         alloc_field(lay, val_id("operation"), alloc_int(op));

         int n = param.coeff_size();
         value coeffs = alloc_array(n);
         for(int i=0;i<n;i++)
             val_array_set_i(coeffs,i,alloc_int(param.coeff(i)));
         alloc_field(lay, val_id("coeffs"), coeffs );
      }
      else if (layer_type=="InnerProduct")
      {
         auto &param = layer.inner_product_param();
         alloc_field(lay, val_id("transpose"), alloc_bool(param.transpose()));
      }

      else
      {
         printf("Unknown layer type %s\n", layer_type.c_str() );
      }


      val_array_push(m,  lay);
   }
}

#endif

value caffeLoad(HxString txtName, HxString binName)
{
   #ifndef NX_CAFFE
   TensorThrow("This binary was built without caffe support");
   return alloc_null();
   #else

   //printf("Caffe -> %s\n", inName.c_str() );

   caffe::NetParameter net;
   value m = alloc_array(0);

   bool loaded = false;
   if (txtName.c_str())
   {
      #ifdef HX_WINDOWS
      int fd = open(txtName.c_str(), _O_RDONLY);
      #else
      int fd = open(txtName.c_str(), O_RDONLY);
      #endif

      if (fd>=0)
      {
         google::protobuf::io::FileInputStream rawstr(fd);
         rawstr.SetCloseOnDelete(true);

         if (!google::protobuf::TextFormat::Parse(&rawstr, &net))
            TensorThrow("Could not parse ptototxt");
         loadCaffeNet(net,m,false);
         loaded = true;
      }
   }
   if (binName.c_str())
   {
      #ifdef HX_WINDOWS
      int fd = open(binName.c_str(), _O_RDONLY | _O_BINARY);
      #else
      int fd = open(binName.c_str(), O_RDONLY);
      #endif

      if (fd>=0)
      {
         google::protobuf::io::FileInputStream rawstr(fd);
         google::protobuf::io::CodedInputStream codedstr(&rawstr);

         rawstr.SetCloseOnDelete(true);
         codedstr.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max() / 2);

         if (!net.ParseFromCodedStream(&codedstr))
            TensorThrow("Could not parse caffemodel");

         loadCaffeNet(net,m,loaded);
         loaded = true;
      }
   }

   if (!loaded)
      TensorThrow("Could not open caffemodel for reading");


   if (net.layers_size()>0)
      TensorThrow("caffemodel version 1 not supported");

   // printf("Input size %d\n", (int)net.input_shape_size() );


   return m;
   #endif
}


DEFINE_PRIME2(caffeLoad);


} // end namespace numerix


