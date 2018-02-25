
#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable

#define READ(x) as_float(intel_sub_group_block_read( (const __global uint *)x))
#define WRITE(x,val) intel_sub_group_block_write( (__global uint *)x, as_uint(val) )

#else

#define READ(x) *(x)
#define WRITE(x,val) *(x) = val

#endif


#if (OP==0)
  #define OPERATOR(a,b) a * b
#elif (OP==1)
  #define OPERATOR(a,b) a + b
#else
  #define OPERATOR(a,b) max(a,b)
#endif


#ifdef cl_intel_subgroups
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Eltwise(const __global float* inSrc0, const __global float *inSrc1, __global float* inDest,
      const int src0Stride, const int src1Stride, const int destStride)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int threadId = get_global_id(2);

   const __global float *src0 = inSrc0 + y*src0Stride + x*CHANNELS + threadId;
   const __global float *src1 = inSrc1 + y*src1Stride + x*CHANNELS + threadId;
   __global float *dest = inDest + y*destStride + x*CHANNELS + threadId;


   for(int channel=0; channel<CHANNELS; channel+=8)
   {
      WRITE( dest + channel, OPERATOR( READ( src0+channel), READ(src1+channel) ) );
   }
}

