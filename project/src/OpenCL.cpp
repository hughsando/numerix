#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <Tensor.h>
#include <Layer.h>
#include <OCL.h>
#include <DynamicLoad.h>


#ifdef HX_WINDOWS
DynamicLibrary openclLib("OpenCL.dll");
#else
#endif


DynamicFunction3(openclLib, CL_API_CALL, cl_int, 0, clGetPlatformIDs, cl_uint, cl_platform_id *,cl_uint *)
DynamicFunction5(openclLib, CL_API_CALL, cl_int, 0, clGetPlatformInfo, cl_platform_id, cl_platform_info, size_t, void *, size_t *)
DynamicFunction5(openclLib, CL_API_CALL, cl_int, 0, clGetDeviceIDs, cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)
DynamicFunction4(openclLib, CL_API_CALL, cl_command_queue, 0, clCreateCommandQueue, cl_context, cl_device_id, cl_command_queue_properties,cl_int *)
typedef void (CL_CALLBACK *ContextCallback)(const char *, const void *, size_t, void *);
DynamicFunction6(openclLib, CL_API_CALL, cl_context, 0, clCreateContext, const cl_context_properties *, cl_uint, const cl_device_id *, ContextCallback, void *, cl_int *)
DynamicFunction5(openclLib, CL_API_CALL, cl_int, 0, clGetDeviceInfo, cl_device_id, cl_device_info, size_t, void *, size_t *)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, 0, clReleaseCommandQueue, cl_command_queue)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, 0, clReleaseContext, cl_context)




namespace numerix
{

OclContext *gOclContext = 0;

void OclContext::setCurrent(OclContext *inContext)
{
   if (inContext)
      inContext->incRef();
   if (gOclContext)
      gOclContext->decRef();

   gOclContext = inContext;
}

bool OclContext::hasCurrent() { return gOclContext; }



class OpenCLContext : public OclContext
{
   int refCount;

   cl_platform_id            platform;
   std::vector<cl_device_id> devices;
   cl_context                context;
   cl_command_queue          queue0;

public:
   OpenCLContext(void *inPlatform, OclDeviceList &inDevices)
   {
      refCount = 1;
      platform = (cl_platform_id)inPlatform;
      devices.resize(inDevices.size());
      for(int i=0;i<devices.size();i++)
         devices[i] = (cl_device_id)inDevices[i];

      const cl_context_properties contextProperties [] =
      {
       CL_CONTEXT_PLATFORM,
       reinterpret_cast<cl_context_properties>(platform),
       0, 0
      };

      cl_int error = 0;
      context = clCreateContext( contextProperties, devices.size(), &devices[0], 0, 0, &error );
      if (!context)
         TensorThrow("Could not create OpenCL Context");


      queue0 = clCreateCommandQueue(context, devices[0], 0, &error);
   }

   ~OpenCLContext()
   {
      clReleaseCommandQueue( queue0 );
      clReleaseContext( context );
   }

   void incRef()
   {
      refCount++;
   }
   void decRef()
   {
      refCount--;
      if (refCount<=0)
         delete this;
   }

};


OclContext *OclContext::create(void *inPlatform, OclDeviceList &inDevices)
{
   return new OpenCLContext(inPlatform, inDevices);
}



OclPlatformList oclGetPlatformList()
{
   cl_uint platformCount = 0;
   clGetPlatformIDs(0, 0, &platformCount);

   std::vector<cl_platform_id> platforms(platformCount);
   if (platformCount)
      clGetPlatformIDs(platformCount, &platforms[0], 0);

   OclPlatformList result;
   for(int i=0;i<platformCount;i++)
      result.push_back( platforms[i] );

   return result;
}


void oclGetPlatformProps(void *inPlatform, OclProps &outProps)
{
   cl_platform_id p = (cl_platform_id)inPlatform;

   if (p)
   {
      const char  *name[] = { "name", "vendor", "version", "profile", "extensions" };
      const cl_platform_info nameId[] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
         CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };

      for(int j=0;j<5;j++)
      {
         size_t infoSize = 0;
         clGetPlatformInfo(p, nameId[j], 0, 0, &infoSize);
         std::vector<char> buf(infoSize+1);
         clGetPlatformInfo(p, nameId[j], infoSize, &buf[0], 0);
         OclProp prop;
         prop.key = name[j];
         prop.value = &buf[0];
         outProps.push_back(prop);
      }
   }
}

OclDeviceList oclGetPlatformDevices(void *inPlatform)
{
   cl_platform_id p = (cl_platform_id)inPlatform;

   OclDeviceList result;

   if (p)
   {
      // get all devices
      cl_uint deviceCount = 0;

      clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, 0, &deviceCount);
      std::vector<cl_device_id> devices(deviceCount);
      if (deviceCount)
         clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, deviceCount, &devices[0], 0);
      for(int i=0;i<deviceCount;i++)
         result.push_back(devices[i]);
   }

   return result;
}

void oclGetDeviceProps(void *inDevice, OclProps &outProps, int &outComputeUnits)
{
   const char  *name[] = { "name", "version", "driverVersion", "deviceOpenCLCVersion" };
   const cl_device_info nameId[] = {
      CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION, CL_DEVICE_OPENCL_C_VERSION };

   for(int j=0;j<4;j++)
   {
      size_t infoSize = 0;
      clGetDeviceInfo( (cl_device_id)inDevice, nameId[j], 0, 0, &infoSize);
      std::vector<char> buf(infoSize+1);
      clGetDeviceInfo((cl_device_id)inDevice, nameId[j], infoSize, &buf[0], 0);
      OclProp prop;
      prop.key = name[j];
      prop.value = &buf[0];
      outProps.push_back(prop);
   }
   cl_uint maxComputeUnits;
   clGetDeviceInfo((cl_device_id)inDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(maxComputeUnits), &maxComputeUnits, 0);


   cl_device_type type = CL_DEVICE_TYPE_DEFAULT;
   clGetDeviceInfo((cl_device_id)inDevice, CL_DEVICE_TYPE, sizeof(type), &type, 0 );

   OclProp prop;
   prop.key="type";
   prop.value="Unknown";
   switch(type)
   {
      case CL_DEVICE_TYPE_CPU: prop.value="CPU"; break;
      case CL_DEVICE_TYPE_GPU: prop.value="GPU"; break;
      case CL_DEVICE_TYPE_ACCELERATOR: prop.value="Accelerator"; break;
      default: ;
   }
   outProps.push_back(prop);

   outComputeUnits = maxComputeUnits;
}






class OpenCLMaxPool : public Layer
{
   int        filterY;
   int        filterX;
   int        strideX;
   int        strideY;

   Padding    padding;
   int        padX;
   int        padY;

public:
   OpenCLMaxPool(int inSizeX, int inSizeY,
           int inStepX, int inStepY,
           Padding inPadding )
   {
      filterX = inSizeX;
      filterY = inSizeY;
      strideX = inStepX;
      strideY = inStepY;
      padding = inPadding;

      padX = padding==padValid ? 0 : ( (filterX-1)/2 );
      padY = padding==padValid ? 0 : ( (filterY-1)/2 );
   }

   ~OpenCLMaxPool()
   {
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("OpenCLMaxPool only supports Float32 tensors");

      CShape &sin = inSrc0->shape;
      if (sin.size()!=3)
         TensorThrow("OpenCLMaxPool only supports H*W*C tensors");

      int srcH = sin[0];
      int srcW = sin[1];
      int channels = sin[2];

      int destW = (srcW + 2*padX - filterX) / strideX + 1;
      int destH = (srcH + 2*padY - filterY) / strideY + 1;

      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape &s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==channels;
      }
      Tensor *result = inBuffer;
      if (!match)
         result = new Tensor( Float32, Shape3(destH, destW, channels ) );


      
      return result;
   }
};

Layer *oclCreateMaxPool(int sizeX, int sizeY,
                        int stepX, int stepY,
                        Padding padding )
{
   return new OpenCLMaxPool(sizeX, sizeY, stepX, stepY, padding);
}






} // end namespace numerix


