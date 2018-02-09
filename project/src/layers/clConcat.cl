
#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Concat(const __global uint* inSrc0, const __global uint *inSrc1, __global uint* inDest)
{
    // threadId is used for source columnn and output feature index
    //const unsigned outBase = get_group_id(2) * 16;
    //const signed int threadId = get_local_id(2);

    const int offset = get_global_id(0);

    __global uint* dest = inDest + offset * (IN0+IN1);

    const __global uint* src0 = inSrc0 + offset * IN0;

    if ( !(IN0&63) )
    {
       for(int i0 = 0; i0<IN0; i0+=64)
          intel_sub_group_block_write4(dest + i0,  intel_sub_group_block_read4(src0 + i0));
    }
    else
    {
       for(int i0 = 0; i0<IN0; i0+=16)
          intel_sub_group_block_write(dest + i0,  intel_sub_group_block_read(src0 + i0));
    }

    dest += IN0;

    const __global uint* src1 = inSrc1 + offset * IN1;

    if ( !(IN1&63))
    {
       for(int i1 = 0; i1<IN1; i1+=64)
          intel_sub_group_block_write4(dest + i1,  intel_sub_group_block_read4(src1 + i1));
    }
    else
    {
       for(int i1 = 0; i1<IN1; i1+=16)
          intel_sub_group_block_write(dest + i1,  intel_sub_group_block_read(src1 + i1));
    }
}
