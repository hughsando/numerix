#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <Tensor.h>
#include <Layer.h>
#include <OCL.h>
#include <DynamicLoad.h>
#include <map>

extern const char *openclConv2D_cl;

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

namespace numerix
{

enum ProgramKey
{
   progMaxPool,
   progConv2DBase,
   progConv2D1x1Base,
   progConv2D_3x3_32,
   progConv2DUnpack,
};



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

   cl_ulong                  localMemory;
   cl_ulong                  computeUnits;

   typedef std::map<ProgramKey, ProgramCache> ProgramMap;
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


      queue0 = clCreateCommandQueue(context, devices[0], 0, &error);
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

   cl_kernel makeKernel(ProgramKey inKey, const char *inProgram, const char *inFunction, const char *buildOptions)
   {
      //printf("makeKernel %s\n", buildOptions);
      ProgramCache &programCache = programMap[inKey];

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
            printf("Error clBuildProgram : %d\n", err);

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





static const char *oclMaxPoolProg = 
"__kernel void MaxPool(const __global float* src, __global float* dest, const int destW, const int destH, const int features, const int srcLastX, int const srcLastY, const int srcShift) {\n"
    "const int x = get_global_id(0);\n"
    "const int y = get_global_id(1);\n"

    "const int srcStride = (destW<<srcShift)*features;\n"
    "const int srcY = y<<srcShift;\n"
    "const int srcX = x<<srcShift;\n"
    "const int dy = srcY<srcLastY ? srcStride : 0;\n"
    "const int dx = srcX<srcLastX ? features : 0;\n"

    "const int o0 = srcY*srcStride + srcX*features;\n"
    "const int o1 = o0+dx;\n"
    "const int o2 = o0+dy;\n"
    "const int o3 = o2+dx;\n"
    "const int dOff = (y*destW+x)*features;\n"
    "for(int f=0;f<features;f++) {\n"
       "float m0 = src[o0+f];\n"
       "float m1 = src[o1+f];\n"
       "float ma = m0>m1 ? m0 : m1;\n"
       "m0 = src[o2+f];\n"
       "m1 = src[o3+f];\n"
       "float mb = m0>m1 ? m0 : m1;\n"
       "dest[dOff+f] = ma>mb ? ma:mb;\n"
    "}\n"
"}"
;



class OpenCLMaxPool : public Layer
{
   int        filterY;
   int        filterX;
   int        strideX;
   int        strideY;

   Padding    padding;
   int        padX;
   int        padY;

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

      padX = padding==padValid ? 0 : ( (filterX-1)/2 );
      padY = padding==padValid ? 0 : ( (filterY-1)/2 );

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLMaxPool - no current ocl context");

      kernel = ctx->makeKernel(progMaxPool, oclMaxPoolProg,"MaxPool", 0);
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


      int destW = 0;
      int destH = 0;
      if (padding==padSame)
      {
         destW = (srcW+strideX-1)/strideX;
         destH = (srcH+strideY-1)/strideY;
      }
      else // padValid
      {
         destW = (srcW-filterX+1 + strideX-1)/strideX;
         destH = (srcH-filterY+1 + strideY-1)/strideY;
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

      const OclData *src = inSrc0->oclRead();

      OclData *dest = result->oclWrite();

      cl_int err = 0;
      err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dest);
      err |= clSetKernelArg(kernel, 2, sizeof(int), &destW);
      err |= clSetKernelArg(kernel, 3, sizeof(int), &destH);
      err |= clSetKernelArg(kernel, 4, sizeof(int), &channels);

      int srcLastX = srcW-1;
      int srcLastY = srcH-1;
      err |= clSetKernelArg(kernel, 5, sizeof(int), &srcLastX);
      err |= clSetKernelArg(kernel, 6, sizeof(int), &srcLastY);
      int srcShift = strideX==1 ? 0 : 1;
      err |= clSetKernelArg(kernel, 7, sizeof(int), &srcShift);

      if (err)
      {
         printf("Error setting kernal arg %d\n", err);
         TensorThrow("OpenCLMaxPool - error setting args");
      }

      if (!ctx)
         TensorThrow("OpenCLMaxPool - no current ocl context");

      size_t globalSize[2] = { (size_t)destW, (size_t)destH };
 
      cl_uint work_dim = 2;
      err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, 0/*Work offset*/, globalSize, 0, 0, NULL, NULL);
      if (err)
         TensorThrow("OpenCLMaxPool - could not clEnqueueNDRangeKernel");

      return result;
   }
};



Layer *oclCreateMaxPool(int sizeX, int sizeY,
                        int stepX, int stepY,
                        Padding padding )
{
   return new OpenCLMaxPool(sizeX, sizeY, stepX, stepY, padding);
}




static const char *oclConv2DProg = 
"__kernel void Conv2D(const __global float* src, const __global float *weights, const __global float *inBias, const int srcW, const int srcH, const int inChannels, __global float *dest, const int outChannels, const int dMin, const int dMax ) {\n"
    "#ifdef EDGES\n"
    "const int id = get_global_id(0);\n"
    "const int x = id<srcW*2 ? id%srcW : id<srcW*2+srcH-2 ? 0 : srcW-1;\n"
    "const int y = id<srcW ? 0 : id<srcW*2 ? srcH-1 : ((id-srcW*2)%(srcH-2)) +1;\n"
    "#else\n"
    "const int x = get_global_id(0);\n"
    "const int y = get_global_id(1);\n"
    "#endif\n"
    "const int srcStride = srcW*inChannels;\n"
    ""
    "if (x<srcW && y<srcH) {\n"
    "int weightOff = 0;\n"
    "int destOff = (y*srcW+x) * outChannels;\n"
    "for(int o=0;o<outChannels;o++) {\n"
       "float sum=inBias[o];\n"
       "for(int dy=dMin; dy<dMax; dy++) {\n"
          "int sy = y+dy;\n"
          "if (sy>=0 && sy<srcH) {\n"
             "for(int dx=dMin; dx<dMax; dx++) {\n"
                 "int sx = x+dx;\n"
                 "if (sx>=0 && sx<srcW) {\n"
                    "int srcOff = sy*srcStride + inChannels*sx;\n"
                    "for(int i=0;i<inChannels;i++)\n"
                        "sum += src[srcOff+i] * weights[weightOff+i];\n"
                 "}\n"
                 "weightOff+=inChannels;\n"
             "}\n"
          "} else {\n"
             "weightOff += inChannels * (dMax-dMin);\n"
          "}\n"
       "}\n"
    "dest[destOff+o] = ACTIVATION(sum);\n"
    "}\n"
   "}\n"
"}";




static const char *oclConv2D1x1 = 
"__kernel void Conv2D(const __global float* inSrc, const __global float *inWeights, const __global float *inBias, const int srcW, const int srcH, const int inChannels, __global float *dest, const int outChannels) {\n"
    "const int x = get_global_id(0);\n"
    "const int y = get_global_id(1);\n"
    "const int srcStride = srcW*inChannels;\n"
    "const int destStride = srcW*outChannels;\n"
    ""
    "int woff = 0;\n"
    "int destOff = y*destStride + outChannels*x;\n"
    "int srcOff = y*srcStride + inChannels*x;\n"
    "for(int o=0;o<outChannels;o++) {\n"
       "float sum=inBias[o];\n"
       "for(int i=0;i<inChannels;i++)\n"
           "sum += inSrc[srcOff+i] * inWeights[woff+i];\n"
       "woff+=inChannels;\n"
    "dest[destOff+o] = ACTIVATION(sum);\n"
    "}"
"}";

class OpenCLConv2D : public Conv2DBase
{
   int padX;
   int padY;
   bool useDmaxDMin;
   bool use32x32;
   cl_kernel kernel;

public:
   OpenCLConv2D(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inActivation, inPadding,  inWeights, 0, inBias)
   {
      kernel = 0;
      useDmaxDMin = true;
      use32x32 = false;
   }

   ~OpenCLConv2D()
   {

   }

   void initKernel()
   {
      int key = 0;
      std::string buildOptions;

      switch(activation)
      {
         case actRelu:
            buildOptions += "-D ACTIVATION(x)=(x<0?0.0f:x)";
            key |= 0x100;
            break;
         case actLeaky:
            buildOptions += "-D ACTIVATION(x)=(x<0?0.1f*x:x)";
            key |= 0x200;
            break;

         default:
            buildOptions += "-D ACTIVATION(x)=(x)";
            break;
      }

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLConv2D - no current ocl context");


      if ( filterX==3 && filterY==3 && !(inputs&31) && !(outputs&31) )
      {
         useDmaxDMin = false;
         use32x32 = true;
         kernel = ctx->makeKernel((ProgramKey)(key|progConv2D_3x3_32), openclConv2D_cl,"Conv2D", buildOptions.c_str());
      }
      else
      {
         key |= is1x1? progConv2D1x1Base : progConv2DBase;
         const char *prog = is1x1 ? oclConv2D1x1 : oclConv2DProg;

         use32x32 = false;
         useDmaxDMin = !is1x1;
         kernel = ctx->makeKernel((ProgramKey)key, prog,"Conv2D", buildOptions.c_str());
      }
   }


   void doRun(Tensor *input, Tensor *output)
   {
      if (!kernel)
         initKernel();

      CShape sin = input->shape;

      const OclData *src = input->oclRead();
      const OclData *w = weights->oclRead();
      const OclData *b = bias->oclRead();
      OclData *dest = output->oclWrite();

      OpenCLContext *ctx = (OpenCLContext *)gOclContext;
      if (!ctx)
         TensorThrow("OpenCLConv2D - no current ocl context");

      cl_int err = 0;
      err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &w);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &b);
      err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &srcW);
      err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &srcH);
      err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &inputs);
      err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &dest);
      err |= clSetKernelArg(kernel, 7, sizeof(cl_int), &outputs);

      if (useDmaxDMin)
      {
         int dmin = -filterX/2;
         int dmax = filterX+dmin;
         err |= clSetKernelArg(kernel, 8, sizeof(cl_int), &dmin);
         err |= clSetKernelArg(kernel, 9, sizeof(cl_int), &dmax);
      }

      if (err)
      {
         printf("Error setting kernal arg %d\n", err);
         TensorThrow("OpenCLConv2D - error setting args");
      }

      if (use32x32)
      {
         /*
         int groupCount  = ctx->computeUnits * 8;
         printf("ideal groupCount %d\n",groupCount);
         int pixelsPerGroup = groupCount<1 ? 32 : (destW*destH/groupCount) & ~31;
         printf("-> pixelsPerGroup = %d\n", pixelsPerGroup);
         if (pixelsPerGroup==0)
            pixelsPerGroup = 32;
         groupCount = (destW*destH + pixelsPerGroup-1) / pixelsPerGroup;
         */

         int groupCount = ctx->computeUnits * 8;
         //printf("ideal groupCount %d\n",groupCount);
         //printf("pixelsPerGroup = %d\n", destW*destH/groupCount );


         size_t globalSize[1] = { (size_t)(32*32*groupCount) };
         size_t localSize[1] = { (size_t)(32*32) };
         cl_uint work_dim = 1;
         size_t *work_offset = 0;
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, localSize, 0, NULL, NULL);
      }
      else
      {
         size_t globalSize[2] = { (size_t)(destW), (size_t)(destH) };
         cl_uint work_dim = 2;
         size_t *work_offset = 0;
         err = clEnqueueNDRangeKernel(ctx->queue0, kernel, work_dim, work_offset, globalSize, 0, 0, NULL, NULL);
      }

      if (err)
         TensorThrow("OpenCLConv2D - could not clEnqueueNDRangeKernel");
      err = clFinish(ctx->queue0);
      if (err)
      {
         printf("Error in clFinish = %d\n", err);
         TensorThrow("OpenCLConv2D - Error waiting clFinish");
      }
   }
};

Layer *oclCreateConv2D(int inStrideY, int inStrideX,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias)
{
   return new OpenCLConv2D(inStrideY, inStrideX, activation, padding, weights, bias);
}








} // end namespace numerix


