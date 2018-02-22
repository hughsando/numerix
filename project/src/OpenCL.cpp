#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <Tensor.h>
#include <Layer.h>
#include <OCL.h>
#include <DynamicLoad.h>
#include <map>
#include <ctype.h>

extern const char *openclConv2D_cl;
extern const char *openclDeconv2D_cl;
extern const char *openclIntelConv2D_cl;
extern const char *openclMaxPool_cl;
extern const char *openclConcat_cl;


// This allows to to profile layers. 
bool gOpenCLProfilingQueue = true;

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
DynamicFunction5(openclLib, CL_API_CALL, cl_int, -99, clGetDeviceInfo, cl_device_id, cl_device_info, size_t, void *, size_t *)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseCommandQueue, cl_command_queue)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseContext, cl_context)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseMemObject, cl_mem)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseProgram, cl_program)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseKernel, cl_kernel)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clReleaseEvent, cl_event)
DynamicFunction1(openclLib, CL_API_CALL, cl_int, -99, clFinish, cl_command_queue)

DynamicFunction5(openclLib, CL_API_CALL, cl_mem, 0, clCreateBuffer, cl_context, cl_mem_flags, size_t, void *, cl_int *)

DynamicFunction9(openclLib, CL_API_CALL, cl_int, 0, clEnqueueWriteBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *)


typedef void (CL_CALLBACK *BuildCallback)(cl_program, void *);
DynamicFunction6(openclLib, CL_API_CALL, cl_int, -1, clBuildProgram, cl_program, cl_uint, const cl_device_id *, const char *, BuildCallback, void *)
DynamicFunction3(openclLib, CL_API_CALL, cl_kernel, 0, clCreateKernel, cl_program, const char *, cl_int *)
DynamicFunction5(openclLib, CL_API_CALL, cl_program, 0, clCreateProgramWithSource, cl_context, cl_uint, const char **, const size_t *, cl_int *)
DynamicFunction9(openclLib, CL_API_CALL, cl_int, -99, clEnqueueReadBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *)
DynamicFunction4(openclLib, CL_API_CALL, cl_int, -99, clSetKernelArg, cl_kernel, cl_uint, size_t, const void *)
DynamicFunction9(openclLib, CL_API_CALL, cl_int, -99, clEnqueueNDRangeKernel, cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *)


DynamicFunction6(openclLib, CL_API_CALL, cl_int, -99, clGetProgramBuildInfo, cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);

DynamicFunction6(openclLib, CL_API_CALL, cl_int, -99, clGetKernelWorkGroupInfo, cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t * );

DynamicFunction2(openclLib, CL_API_CALL, cl_int, -99, clWaitForEvents, cl_uint, const cl_event *);

DynamicFunction5(openclLib, CL_API_CALL, cl_int, -99, clGetEventProfilingInfo, cl_event, cl_profiling_info, size_t, void *,size_t *);



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


struct ProgramCache
{
   cl_program program;
   cl_kernel  kernel;

   ProgramCache()
   {
      program = 0;
      kernel = 0;
   }
   ProgramCache(const ProgramCache &p) : program(p.program), kernel(p.kernel) { }
   void operator=(const ProgramCache &p) { program = p.program; kernel=p.kernel; }

   void destroy()
   {
      if (kernel)
         clReleaseKernel(  kernel );
      if (program)
         clReleaseProgram( program );
   }
};

class OpenCLContext : public OclContext
{
public:
   int refCount;

   cl_platform_id            platform;
   std::vector<cl_device_id> devices;
   cl_context                context;
   cl_command_queue          queue0;
   bool                      useIntelMethod;

   cl_ulong                  localMemory;
   cl_ulong                  computeUnits;

   typedef std::map<unsigned int, ProgramCache> ProgramMap;
   ProgramMap programMap;

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
      context = clCreateContext( contextProperties, devices.size(), &devices[0], SOnCLInfo, this, &error );
      if (!context || error)
         TensorThrow("Could not create OpenCL Context");


      char buf[1024];
      clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 1024, buf, 0);
      for(int i=0;i<5;i++)
         buf[i] = tolower(buf[i]);
      buf[5]='\0';
      useIntelMethod = std::string(buf)=="intel";

      queue0 = clCreateCommandQueue(context, devices[0], gOpenCLProfilingQueue ? CL_QUEUE_PROFILING_ENABLE : 0, &error);
      if (!queue0 || error)
         TensorThrow("Could not create OpenCL command queue");

      localMemory = 0;
      clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemory, 0);
      computeUnits = 0;
      clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &computeUnits, 0);
      /*
      printf("Local memory :==================== %d\n", (int)localMemory );
      printf("Compue units :==================== %d\n", (int)computeUnits );
      cl_ulong wgs = 0;
      clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &wgs, 0);
      printf("WGS :==================== %d\n", (int)wgs );
      */

      setCurrent(this);
   }

   ~OpenCLContext()
   {
      for(ProgramMap::iterator i=programMap.begin();i!=programMap.end();++i)
         i->second.destroy();

      if (queue0)
         clReleaseCommandQueue( queue0 );
      if (context)
         clReleaseContext( context );
   }

   static void SOnCLInfo(const char *errinfo, const void *private_info, size_t cb, void *user_data)
   {
      ((OpenCLContext *)user_data)->onCLInfo(errinfo, private_info, cb);
   }

   void onCLInfo(const char *errinfo, const void *private_info, size_t cb)
   {
      printf("############################## onCLInfo -> %s\n", errinfo);
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

   cl_kernel makeKernel(const char *inMethod, const char *inProgram, const char *inFunction, std::string inOps = "")
   {
      inOps += " -D ";
      inOps += inMethod;
      if (useIntelMethod)
         inOps += " -D INTEL";
      inOps += " -Werror";
      const char *buildOptions = inOps.c_str();

      unsigned int hash = 0;
      for(int i=0;buildOptions[i];i++)
         hash = hash*223 + ((const unsigned char *)buildOptions)[i];

      printf("makeKernel %s\n", buildOptions);
      ProgramCache &programCache = programMap[hash];

      if (!programCache.kernel)
      {
         cl_int err = 0;
         cl_program prog = clCreateProgramWithSource(context, 1, (const char **) &inProgram, 0, &err);
         if (err || !prog)
         {
            printf("Error in clCreateProgramWithSource : %d\n", err);
            TensorThrow("clCreateProgramWithSource - error");
         }

         err = clBuildProgram(prog, 1, &devices[0], buildOptions, 0, 0);
         if (err)
         {
            printf("Error clBuildProgram : (%p) %d\n", devices[0], err);

            size_t size = 0;
            clGetProgramBuildInfo(prog, devices[0], CL_PROGRAM_BUILD_LOG, 0, 0, &size);
            if (size>0)
            {
               std::vector<char> buffer(size+1);
               clGetProgramBuildInfo(prog, devices[0], CL_PROGRAM_BUILD_LOG, size, &buffer[0], 0);
               printf("  : %s\n", &buffer[0]);
            }

            TensorThrow("clBuildProgram - error");
         }


         err = 0;
         cl_kernel kernel = clCreateKernel(prog, inFunction, &err);
         if (err || !kernel)
         {
            printf("Error clCreateKernel : %d\n", err);
            TensorThrow("clCreateKernel - error");
         }

         programCache.kernel = kernel;
         /*
         printf("makeKernel %s\n", buildOptions);

         cl_ulong val = 0;
         clGetKernelWorkGroupInfo(kernel, devices[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(cl_ulong), &val, 0);
         printf("  multiple of -> %d\n", (int)val);
         val = 0;
         clGetKernelWorkGroupInfo(kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(cl_ulong), &val, 0);
         printf("  wgs  of -> %d\n", (int)val);
         */
      }

      return programCache.kernel;
   }
};




OclData *oclAlloc(int size)
{
   OpenCLContext *ctx = (OpenCLContext *)gOclContext;
   if (!ctx)
      TensorThrow("oclAlloc - no current ocl context");

   cl_int err = 0;
   cl_mem buffer = clCreateBuffer( ctx->context, CL_MEM_READ_WRITE, size, 0, &err );
   if (err)
      TensorThrow("oclAlloc - could not create buffer");

   return buffer;
}

void oclFree(OclData *inBuffer)
{
   printf("Release mem %p\n", inBuffer);
   if (clReleaseMemObject(inBuffer))
      TensorThrow("oclFree - could not clReleaseMemObject");
}

void oclDownload(unsigned char *buffer, const OclData *inData, int n)
{
   OpenCLContext *ctx = (OpenCLContext *)gOclContext;
   if (!ctx)
      TensorThrow("oclAlloc - no current ocl context");

    if (clFinish(ctx->queue0))
       TensorThrow("oclDownload - error finishing");
 
    // Read the results from the device
    if (clEnqueueReadBuffer(ctx->queue0, (cl_mem)inData, CL_TRUE, 0, n, buffer, 0, 0, 0 ))
       TensorThrow("oclDownload - could not clEnqueueReadBuffer");
}


void oclUpload(OclData *buffer, const unsigned char *inData, int n)
{
   OpenCLContext *ctx = (OpenCLContext *)gOclContext;
   if (!ctx)
      TensorThrow("oclUpload - no current ocl context");

   bool blocking = false;
   cl_int err =  clEnqueueWriteBuffer(ctx->queue0, buffer, blocking, 0, n, inData, 0, NULL, NULL);
   if (err)
   {
      printf("Error in clEnqueueWriteBuffer %d\n", err);
      TensorThrow("oclUpload - could not clEnqueueWriteBuffer");
   }
}



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


class OpenCLLayerData
{
public:
   cl_event timingEvent;
   double   totalTime;
   int      runCount;
   bool     waiting;


   OpenCLLayerData()
   {
      waiting = false;
      totalTime = 0.0;
      runCount = 0;
      timingEvent = 0;
   }

   ~OpenCLLayerData()
   {
      updateStats();
   }

   void updateStats()
   {
      if (waiting)
      {
         waiting = false;
         clWaitForEvents(1, &timingEvent);

         cl_ulong time_start;
         cl_ulong time_end;

         clGetEventProfilingInfo(timingEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
         clGetEventProfilingInfo(timingEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

         clReleaseEvent(timingEvent);

         double nanoSeconds = time_end-time_start;
         totalTime += nanoSeconds * 1e-9;
         runCount++;
      }
   }

   cl_event *startKernel()
   {
      updateStats();
      if (Layer::openCLTimingEvents)
      {
         waiting = true;
         return &timingEvent;
      }
      return 0;
   }

   double getRunTime()
   {
      updateStats();
      if (runCount==0)
         return 0.0;
      return totalTime/runCount;
   }


};

class OpenCLLayer : public Layer
{
public:
   OpenCLLayerData data;

   cl_event *startKernel() { return data.startKernel(); }
   void updateStats() { data.updateStats(); }
   double getRunTime() { return data.getRunTime(); }

};




// --- MaxPool --------------------------------------



class OpenCLMaxPool : public OpenCLLayer
{
   int        filterY;
   int        filterX;
   int        strideX;
   int        strideY;

   Padding    padding;
   int        padX;
   int        padY;
   int        lastChannels;

   bool       odd;

   cl_kernel  kernel;


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

      padX = padding.type==Padding::padValid ? 0 : ( (filterX-1)/2 );
      padY = padding.type==Padding::padValid ? 0 : ( (filterY-1)/2 );

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLMaxPool - no current ocl context");

      lastChannels = 0;
      kernel = 0;
   }

   ~OpenCLMaxPool()
   {
   }

   void initKernel(OpenCLContext *ctx, int channels)
   {
      lastChannels = channels;
      char buildOptions[1024];
      sprintf(buildOptions," -D CHANNELS=%d", channels);

      if (filterX==2 && filterY==2)
         kernel = ctx->makeKernel("MAX_POOL", openclMaxPool_cl,"MaxPool2x2", buildOptions);
      else if (filterX==3 && filterY==3)
         kernel = ctx->makeKernel("MAX_POOL", openclMaxPool_cl,"MaxPool3x3", buildOptions);
      else
         TensorThrow("TODO - unimplemented OpenCL MaxPool size");
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


      int destW = 0;
      int destH = 0;
      
      if (padding.type==Padding::padSame)
      {
         destW = (srcW+strideX-1)/strideX;
         destH = (srcH+strideY-1)/strideY;
      }
      else // padValid
      {
         destW = (srcW)/strideX;
         destH = (srcH)/strideY;
      }


      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape &s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==channels;
      }
      Tensor *result = inBuffer;
      if (!match)
         result = new Tensor( Float32, Shape3(destH, destW, channels ) );


      OpenCLContext *ctx = (OpenCLContext *)gOclContext;

      if (kernel && lastChannels!=channels)
         kernel = 0;
      if (!kernel)
         initKernel(ctx, channels);


      const OclData *src = inSrc0->oclRead();

      OclData *dest = result->oclWrite();

      cl_int err = 0;
      err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dest);
      err |= clSetKernelArg(kernel, 2, sizeof(int), &destW);
      err |= clSetKernelArg(kernel, 3, sizeof(int), &destH);

      int srcLastX = srcW-1;
      int srcLastY = srcH-1;
      err |= clSetKernelArg(kernel, 4, sizeof(int), &srcLastX);
      err |= clSetKernelArg(kernel, 5, sizeof(int), &srcLastY);
      int srcShift = strideX==1 ? 0 : 1;
      err |= clSetKernelArg(kernel, 6, sizeof(int), &srcShift);
      int srcStride = srcW * channels;
      err |= clSetKernelArg(kernel, 7, sizeof(int), &srcStride);

      if (err)
      {
         printf("Error setting kernal arg %d\n", err);
         TensorThrow("OpenCLMaxPool - error setting args");
      }

      if (!ctx)
         TensorThrow("OpenCLMaxPool - no current ocl context");



      

      size_t globalSize[2] = { (size_t)destW, (size_t)destH };
 
      cl_uint work_dim = 2;
      err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, 0, globalSize, 0, 0, NULL, startKernel());
      if (err)
         TensorThrow("OpenCLMaxPool - could not clEnqueueNDRangeKernel");


      if (Layer::accurateTimes)
      {
         err = clFinish(ctx->queue0);
         if (err)
         {
            printf("Error in clFinish = %d\n", err);
            TensorThrow("OpenCLConv2D - Error waiting clFinish");
         }
      }


      return result;
   }
};


Layer *oclCreateMaxPool(int sizeX, int sizeY,
                        int stepX, int stepY,
                        Padding padding )
{
   return new OpenCLMaxPool(sizeX, sizeY, stepX, stepY, padding);
}




// --- Concat --------------------------------------

static const char *oclConcatProg = 
"__kernel void Concat(const __global float* src0, const __global float *src1, __global float* dest, const int srcW, const int srcH) {\n"
    "const int x = get_global_id(0);\n"
    "const int y = get_global_id(1);\n"
    "__global float *d = dest + (y*srcW + x)*(IN0+IN1);\n"

    "if (IN0&15) {\n"
       "const __global float *i0 = src0 + (y*srcW + x)*IN0;\n"
       "for(int f=0;f<IN0;f++) {\n"
          "d[f] = i0[f];\n"
       "}\n"
     "} else {\n"
       "const __global float4 *i0 = (const __global float4 *)(src0 + (y*srcW + x)*IN0);\n"
       "__global float4 *d4 = (__global float4 *)(d);\n"
       "for(int f=0;f<IN0;f+=16) {\n"
          "d4[0] = i0[0];\n"
          "d4[1] = i0[1];\n"
          "d4[2] = i0[2];\n"
          "d4[3] = i0[3];\n"
          "i0+=4;\n"
          "d4+=4;\n"
       "}\n"
     "}\n"
    "d+=IN0;\n"

    "if (IN1&15) {\n"
       "const __global float *i1 = src1 + (y*srcW + x)*IN1;\n"
       "for(int f=0;f<IN0;f++) {\n"
          "d[f] = i1[f];\n"
       "}\n"
     "} else {\n"
       "const __global float4 *i1 = (const __global float4 *)(src1 + (y*srcW + x)*IN1);\n"
       "__global float4 *d4 = (__global float4 *)(d);\n"
       "for(int f=0;f<IN1;f+=16) {\n"
          "d4[0] = i1[0];\n"
          "d4[1] = i1[1];\n"
          "d4[2] = i1[2];\n"
          "d4[3] = i1[3];\n"
          "i1+=4;\n"
          "d4+=4;\n"
       "}\n"
     "}\n"
"}"
;



class OpenCLConcat : public OpenCLLayer
{
   cl_kernel  kernel;
   int lastIn0;
   int lastIn1;
   bool intelMethod;

public:
   OpenCLConcat( )
   {
      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLConcat - no current ocl context");

      lastIn0 = lastIn1 = 0;
      kernel = 0;
      intelMethod = false;
   }

   ~OpenCLConcat()
   {
   }

   void initKernel(OpenCLContext *ctx, int in0, int in1)
   {
      lastIn0 = in0;
      lastIn1 = in1;
      char buildOptions[1024];
      sprintf(buildOptions," -DIN0=%d -DIN1=%d", in0, in1);
      intelMethod = ctx->useIntelMethod && !(in0 & 15) && !(in1 & 15);
      const char *prog = intelMethod ? openclConcat_cl : oclConcatProg;
      kernel = ctx->makeKernel("CONCAT", prog,"Concat", buildOptions);
   }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer)
   {

      if (inSrc0->type != inSrc1->type || inSrc0->type!=Float32)
         TensorThrow("Concat - input types must be float32");

      CShape sin0 = inSrc0->shape;
      CShape sin1 = inSrc1->shape;
      if (sin0.size()!=3 || sin1.size()!=3)
         TensorThrow("Concat only supports H*W*C tensors");

      if (sin0[0]!=sin1[0] || sin0[1]!=sin1[1])
         TensorThrow("Concat - mismatch image sizes");

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("Concat - no OpenCL context");

      int srcH = sin0[0];
      int srcW = sin0[1];
      int c0 = sin0[2];
      int c1 = sin1[2];

      if (kernel && lastIn0!=c0 || lastIn1!=c1)
         kernel = 0;
      if (!kernel)
         initKernel(ctx, c0,c1);


      int channels = c0+c1;

      Tensor *result = Tensor::makeBuffer(inBuffer, srcW, srcH, channels, inSrc0->type);


      const OclData *src0 = inSrc0->oclRead();
      const OclData *src1 = inSrc1->oclRead();

      OclData *dest = result->oclWrite();

      cl_int err = 0;
      err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src0);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &src1);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dest);
      if (!intelMethod)
      {
         err |= clSetKernelArg(kernel, 3, sizeof(int), &srcW);
         err |= clSetKernelArg(kernel, 4, sizeof(int), &srcH);
      }

      if (err)
      {
         printf("Error setting kernal arg %d\n", err);
         TensorThrow("OpenCLConcat - error setting args");
      }

      if (!ctx)
         TensorThrow("OpenCLConcat - no current ocl context");

      if (intelMethod)
      {
         size_t globalSize[2] = { (size_t)(srcW*srcH), 16 };
         size_t localSize[2] = { 1, 16 };
         cl_uint work_dim = 2;
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, 0/*Work offset*/, globalSize, localSize, 0, NULL, startKernel());
         if (err)
            TensorThrow("OpenCLConcat - could not clEnqueueNDRangeKernel");
      }
      else
      {
         size_t globalSize[2] = { (size_t)srcW, (size_t)srcH };
 
         cl_uint work_dim = 2;
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, 0/*Work offset*/, globalSize, 0, 0, NULL, startKernel());
         if (err)
            TensorThrow("OpenCLConcat - could not clEnqueueNDRangeKernel");
      }

      if (Layer::accurateTimes)
      {
         err = clFinish(ctx->queue0);
         if (err)
         {
            printf("Error in clFinish = %d\n", err);
            TensorThrow("OpenCLConv2D - Error waiting clFinish");
         }
      }

      return result;
   }

};



Layer *oclCreateConcat( )
{
   return new OpenCLConcat( );
}


// --- Conv2D --------------------------------------------------------------------



class OpenCLConv2D : public Conv2DBase
{
   int padX;
   int padY;
   bool useTiled3x3;
   bool useTiled1x1;
   bool useIntelTiled1x1;
   bool useIntelTiled3x3;
   bool useIntelTiled3x3x3;
   bool useIntelDeconv;
   int  threads;
   cl_kernel kernel;
   Shape kernelShape;
   OpenCLLayerData data;
   Tensor *overrideWeights;
   bool group8;

   // For deconv
   int srcX0;
   int srcY0;
   int srcX1;
   int srcY1;
   int srcTx;
   int srcTy;



public:
   OpenCLConv2D(int inStrideY, int inStrideX, bool inIsDeconvolution,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inIsDeconvolution,inActivation, inPadding,  inWeights, 0, inBias)
   {
      kernel = 0;
      useTiled3x3 = false;
      useTiled1x1 = false;
      useIntelTiled1x1 = false;
      useIntelTiled3x3 = false;
      useIntelTiled3x3x3 = false;
      useIntelDeconv = false;
      threads = 16;
      overrideWeights = 0;
      group8 = false;

      padX = padding.type==Padding::padValid ? 0 : ( (filterX-1)/2 );
      padY = padding.type==Padding::padValid ? 0 : ( (filterY-1)/2 );
   }

   ~OpenCLConv2D()
   {
      if (overrideWeights)
         overrideWeights->decRef();
   }

   cl_event *startKernel() { return data.startKernel(); }
   void updateStats() { data.updateStats(); }
   double getRunTime() { return data.getRunTime(); }

   void initKernel()
   {
      std::string buildOptions;

      switch(activation)
      {
         case actRelu:
            buildOptions += "-D ACTIVATION(x)=fmax(x,0.0f)";
            break;
         case actLeaky:
            buildOptions += "-D ACTIVATION(x)=fmax(x,0.1f*x)";
            break;

         default:
            buildOptions += "-D ACTIVATION(x)=(x)";
            break;
      }
      char argBuf[1000];
      int dMin = padding.type==Padding::padValid ? 0 : (-filterX)/2;
      int dMax = dMin + filterX;


      if ( !(outputs&63) )
         threads = 32;
      else if (!(outputs&31) )
         threads = 16;
      else if (!(outputs&15) )
         threads = 8;
      else if (!(outputs&7) )
         threads = 4;
      else
         threads = 0;


      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLConv2D - no current ocl context");

      bool wasI3x3 = useIntelTiled3x3;
      bool wasI3x3x3 = useIntelTiled3x3x3;

      useTiled1x1 = is1x1 && threads;
      useIntelTiled1x1 = is1x1 && (threads>=8) && !(outputs&15) && ctx->useIntelMethod;
      useTiled3x3 = strideX==1 && strideY==1 && filterX==3 && filterY==3 && !(inputs&3) && threads;
      useIntelTiled3x3x3 = filterX==3 && filterY==3 && inputs==3 && !(outputs&15) && strideX==2 && strideY==2 && ctx->useIntelMethod;

      //printf("useIntelTiled3x3 ----- %d:  %d %d %d %d %d %d %d\n", useIntelTiled3x3x3,
      //       filterX==3, filterY==3, inputs==3, threads, strideX==2, strideY==2, ctx->useIntelMethod );

      useIntelTiled3x3 = strideX==1 && strideY==1 && filterX==3 && filterY==3 && !(inputs&7) && !(outputs&7) && ctx->useIntelMethod;


      if ( (!wasI3x3 && useIntelTiled3x3) || (!wasI3x3x3 && useIntelTiled3x3x3) || isDeconvolution )
         rebuildWeights();


      if (isDeconvolution)
      {
         int SW = filterX/strideX;
         int SH = filterY/strideY;

         int srcX0 = ((padOx) >> strideShiftX) - SW/2;
         int srcX1 = ((destW+padOx + strideX-1)>>strideShiftX) - SW/2;


         int srcY0 = ((padOy) >> strideShiftY) - SH/2;
         int srcY1 = ((destH+padOy + strideX-1)>>strideShiftY) - SH/2;

         srcTx = srcX1 - srcX0;
         srcTy = srcY1 - srcY0;


         // Iterate over src
         sprintf(argBuf," -D DEST_W=%d -D DEST_H=%d -D INPUTS=%d -D OUTPUTS=%d",
                 destW, destH, inputs, outputs);
         buildOptions += argBuf;
         sprintf(argBuf," -D SRC_W=%d -D SRC_H=%d",
                 srcW, srcH, inputs, outputs);
         buildOptions += argBuf;
         sprintf(argBuf," -D FILTER_X=%d -D FILTER_Y=%d -D SHIFT_X=%d -D SHIFT_Y=%d",
                 filterX, filterY, strideShiftX, strideShiftY);
         buildOptions += argBuf;
         sprintf(argBuf," -D PAD_X=%d -D PAD_Y=%d -D SRC_TX=%d -D SRC_TY=%d", padOx, padOy, srcTx, srcTy);
         buildOptions += argBuf;

         const char *prog = openclDeconv2D_cl;
         if (!group8)
            kernel = ctx->makeKernel("ODD_DECONV2D", prog, "Deconv2D", buildOptions);
         else if (ctx->useIntelMethod)
            kernel = ctx->makeKernel("INTEL_DECONV2D", prog, "Deconv2D", buildOptions);
         else
            kernel = ctx->makeKernel("DECONV2D", prog, "Deconv2D", buildOptions);
      }
      else if (useIntelTiled1x1 || useIntelTiled3x3 || useIntelTiled3x3x3)
      {
         sprintf(argBuf," -D INPUT0_SIZE_X=%d -D INPUT0_SIZE_Y=%d -D INPUT0_FEATURE_NUM=%d ",
                 srcW, srcH, inputs);
         buildOptions += argBuf;
         sprintf(argBuf," -D OUTPUT_SIZE_X=%d -D OUTPUT_SIZE_Y=%d -D OUTPUT_FEATURE_NUM=%d",
                 destW, destH, outputs);
         buildOptions += argBuf;
         sprintf(argBuf," -D FILTER_OFM_PITCH=%d -D FILTER_IFM_NUM=%d ",
                 inputs, inputs);
         sprintf(argBuf," -D PADX=%d -D PADY=%d ", padX, padY);
         buildOptions += argBuf;

         const char *prog = openclIntelConv2D_cl;

         if (useIntelTiled1x1)
            kernel = ctx->makeKernel("INTEL_TILED_1X1", prog, "Conv2D_1x1", buildOptions);
         else if (useIntelTiled3x3x3)
            kernel = ctx->makeKernel("INTEL_TILED_3X3X3", prog, "Conv2D_3X3X3", buildOptions);
         else
            kernel = ctx->makeKernel("INTEL_TILED_3X3", prog, "Conv2D_3x3", buildOptions);
      }
      else
      {
         sprintf(argBuf," -D INPUTS=%d -D OUTPUTS=%d -D FX=%d -D FY=%d -D STRIDE_X=%d -D STRIDE_Y=%d -D DEST_W=%d -D DEST_H=%d -D dMin=%d -D dMax=%d -D OVEC=%d",inputs,outputs,filterX,filterY, strideX, strideY, destW, destH, dMin, dMax, threads);
         buildOptions += argBuf;

         if ( useTiled3x3 )
         {
            buildOptions += argBuf;
            kernel = ctx->makeKernel("CONV2D_3x3", openclConv2D_cl,"Conv2D", buildOptions);
         }
         else
         {
            const char *func = is1x1 ? (useTiled1x1 ? "TILED_1X1" : "CONV2D_1x1") : "CONV2D_SIMPLE";

            const char *prog = openclConv2D_cl;
            kernel = ctx->makeKernel(func, prog, "Conv2D", buildOptions);
         }
      }
   }


   void doRun(Tensor *input, Tensor *output)
   {
      if (kernel && kernelShape!=input->shape)
         kernel = 0;

      if (!kernel)
      {
         kernelShape = input->shape;
         initKernel();
      }

      CShape sin = input->shape;

      const OclData *src = input->oclRead();
      const OclData *w = (overrideWeights ? overrideWeights : weights)->oclRead();
      const OclData *b = bias->oclRead();
      OclData *dest = output->oclWrite();

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLConv2D - no current ocl context");

      cl_int err = 0;
      err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dest);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &w);
      err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &b);
      if (!useIntelTiled1x1 && !useIntelTiled3x3 && !useIntelTiled3x3x3 && !isDeconvolution)
      {
         err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &srcW);
         err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &srcH);
      }

      if (err)
      {
         printf("Error setting kernal arg %d\n", err);
         TensorThrow("OpenCLConv2D - error setting args");
      }

      size_t *work_offset = 0;

      if (isDeconvolution)
      {
         if (group8)
         {
            cl_uint work_dim = 3;
            size_t globalSize[3] = { (size_t)(srcTx*srcTy), (size_t)outputs/8, (size_t)8  };
            size_t localSize[3] = { 1, 1, 8 };
            err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());
         }
         else
         {
            cl_uint work_dim = 1;
            size_t globalSize[1] = { (size_t)(srcTx*srcTy) };
            size_t *localSize = 0;
            err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());
         }
      }
      else if (useIntelTiled1x1)
      {
         int groupCount = srcH;
         cl_uint work_dim = 2;
         size_t globalSize[2] = { (size_t)((destW*destH+15)/16),  (size_t)(outputs)  };
         size_t localSize[2] = { (size_t)(1),  (size_t)(16)  };
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());
      }
      else if (useIntelTiled3x3x3)
      {
         int groupCount = srcH;
         cl_uint work_dim = 3;
         size_t xGroups = (destW+6)/7;
         size_t yGroups = (destH+7)/8;
         size_t outputGroups = outputs/16;

         size_t localSize[3] = { 1,  1, 16 };
         size_t globalSize[3] = { xGroups*localSize[0], yGroups*localSize[1], outputGroups*localSize[2] };

         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());
      }
      else if (useIntelTiled3x3)
      {
         int groupCount = srcH;
         cl_uint work_dim = 3;
         size_t xGroups = (destW+3)/4;
         size_t yGroups = (destH+3)/4;
         size_t outputGroups = outputs/8;

         size_t localSize[3] = { 1,  1,  8 };
         size_t globalSize[3] = { xGroups*localSize[0], yGroups*localSize[1], outputGroups*localSize[2] };

         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());
      }
      else if (useTiled3x3 || useTiled1x1)
      {
         int groupCount = srcH;
         cl_uint work_dim = 2;
         size_t globalSize[2] = { (size_t)threads,  (size_t)groupCount  };
         size_t localSize[2] = {  (size_t)threads, 1 };
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, startKernel());

      }
      else
      {
         size_t globalSize[2] = { (size_t)(destW), (size_t)(destH) };
         cl_uint work_dim = 2;
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, 0, 0, NULL, startKernel());
      }

      if (err)
         TensorThrow("OpenCLConv2D - could not clEnqueueNDRangeKernel");

      if (Layer::accurateTimes)
      {
         err = clFinish(ctx->queue0);
         if (err)
         {
            printf("Error in clFinish = %d\n", err);
            TensorThrow("OpenCLConv2D - Error waiting clFinish");
         }
      }
   }

   void rebuildWeights()
   {
      Conv2DBase::rebuildWeights();
      if (overrideWeights)
      {
         overrideWeights->decRef();
         overrideWeights = 0;
      }

      if (isDeconvolution)
      {
         Shape shape = weights->shape;
         int outs = shape[0];
         int h = shape[1];
         int w = shape[2];
         int ins = shape[3];
         overrideWeights = new Tensor( weights->type, shape );

         int xOff = filterX/2;
         int yOff = filterY/2;

         int SW = filterX/strideX;
         int SH = filterY/strideY;

         int idx = 0;
         group8 =  ( !(ins&0x7) && !(outs&0x7) );
         if (group8)
         {
            for(int oBase=0;oBase<outs;oBase+=8)
               for(int iBase=0; iBase<ins; iBase+=8)
                  for(int dy=0;dy<strideY;dy++)
                     for(int dx=0;dx<strideX;dx++)
                        for(int sy=0;sy<SH;sy++)
                           for(int sx=0;sx<SW;sx++)
                           {
                              int wy = (dy+yOff+sy*strideY) % filterY;
                              int wx = (dx+xOff+sx*strideX) % filterX;

                              for(int o=0;o<8;o++)
                                 for(int i=0;i<8;i++)
                                    overrideWeights->setFloatAt( idx++, weights->getFloat(oBase+o,wy,wx,iBase+i) );
                           }
         }
         else
         {
            for(int dy=0;dy<strideY;dy++)
               for(int dx=0;dx<strideX;dx++)
                  for(int o=0;o<outs;o++)
                     for(int i=0;i<ins;i++)
                        for(int sy=0;sy<SH;sy++)
                           for(int sx=0;sx<SW;sx++)
                           {
                              int wy = (dy+yOff+sy*strideY) % filterY;
                              int wx = (dx+xOff+sx*strideX) % filterX;
 
                              overrideWeights->setFloatAt( idx++, weights->getFloat(o,wy,wx,i) );
                           }
         }
      }
      else if (useIntelTiled3x3 || useIntelTiled3x3x3)
      {
         Shape shape = weights->shape;
         int outs = shape[0];
         int h = shape[1];
         int w = shape[2];
         int ins = shape[3];
         overrideWeights = new Tensor( weights->type, shape );

         int outputGroups = useIntelTiled3x3x3 ? 16 : 8;

         int idx = 0;
         int inputs = std::min(8,ins);
         for(int oBase=0;oBase<outs;oBase+=outputGroups)
            for(int iBase=0; iBase<ins; iBase+=8)
                for(int y=0;y<h;y++)
                   for(int x=0;x<w;x++)
                      for(int i=0;i<inputs;i++)
                         for(int o=0;o<outputGroups;o++)
                            overrideWeights->setFloatAt( idx++, weights->getFloat(oBase+o,y,x,iBase+i) );
      }
   }


};

Layer *oclCreateConv2D(int inStrideY, int inStrideX, bool inIsDeconvolution,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias)
{
   return new OpenCLConv2D(inStrideY, inStrideX, inIsDeconvolution, activation, padding, weights, bias);
}








} // end namespace numerix


