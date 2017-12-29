#include <Tensor.h>
#include <Layer.h>
#include <stdio.h>
#include <stdexcept>
#include <algorithm>
#include <Ops.h>
#include "mvnc.h"

namespace numerix
{

class Movidius : public Layer
{
   void *deviceHandle;
   void *graphHandle;
   Shape outputShape;
   std::vector<short> f16buf;

public:
   Movidius(const std::string &inDeviceName, const unsigned char *graphData, size_t dataLength, CShape inOutputSize)
   {
      outputShape = inOutputSize;

      deviceHandle = 0;
      graphHandle = 0;

      // Try to open the NCS device via the device name
      mvncStatus retCode = nxMvncOpenDevice(inDeviceName.c_str(), &deviceHandle);
      if (retCode != MVNC_OK)
      { 
         char buf[1024];
         sprintf(buf, " - Could not open NCS device :%d.", retCode);
         TensorThrow( (inDeviceName + buf).c_str() );
      }


      // allocate the graph
       retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphData, dataLength);
       if (retCode != MVNC_OK)
       {
          char buf[1024];
          sprintf(buf, " - Could not allocate graph :%d.", retCode);
          TensorThrow( (inDeviceName + buf).c_str() );
       }

      // deviceHandle is ready to use now.  
      // Pass it to other NC API calls as needed and close it when finished.
      printf("NCS '%s' Device opened normally.\n", inDeviceName.c_str());
   }


   ~Movidius()
   {
      if (graphHandle)
      {
          mvncDeallocateGraph(graphHandle);
      }

      if (deviceHandle)
      {
         mvncStatus retCode = mvncCloseDevice(deviceHandle);
         if (retCode!=MVNC_OK)
         {
            printf("Could not close NCS devide: %d\n", retCode);
         }
      }
   }



   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      int destChannels = 1;
      int destH = 1;
      int destW = 1;

      if (outputShape.size()==1)
      {
         destW = outputShape[0];
      }
      else if (outputShape.size()==2)
      {
         destH = outputShape[0];
         destW = outputShape[1];
      }
      else if (outputShape.size()==3)
      {
         destH = outputShape[0];
         destW = outputShape[1];
         destChannels = outputShape[2];
      }
      else
      {
         TensorThrow("Movidius - could not use output shape");
      }

      if (inSrc0->type != Float32)
         TensorThrow("Movidius input should be float32");

      Tensor *result = Tensor::makeBuffer(inBuffer, destW, destH, destChannels, Float32);

      int elems = inSrc0->elementCount;
      f16buf.resize(elems);

      floattofp16((u8*)&f16buf[0], (const float *)inSrc0->cpuRead(), elems);

       // start the inference with mvncLoadTensor()
      mvncStatus retCode = mvncLoadTensor(graphHandle, &f16buf[0], elems*sizeof(short), 0);
      if (retCode != MVNC_OK)
         TensorThrow("Movidius - could not load tensor");

      void* resultData16 = 0;
      void* userParam = 0;
      unsigned int lenResultData = 0;

      retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
      if (retCode != MVNC_OK)
         TensorThrow("Movidius - could not get result");

      int resultCount = lenResultData/sizeof(short);
      if (resultCount!=destW*destH*destChannels)
      {
         char buf[1024];
         sprintf(buf, "Movidius - mismatch output size %dx%dx%d (%d) != %d", destW, destH, destChannels, destW*destH*destChannels, resultCount);
         TensorThrow(buf);
      }

      fp16tofloat( (float *)result->cpuWrite(), (const u8 *)resultData16, resultCount );

      return result;
   }

};

Layer *Layer::createMovidius(const std::string &inDeviceName, const unsigned char *graphData, size_t dataLength, CShape outputShape)
{
   return new Movidius(inDeviceName, graphData, dataLength, outputShape);
}

} // end namespace numerix

