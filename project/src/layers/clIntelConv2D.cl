
// Copyright (c) 2016-2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define __CAT_FUNC(x, y) FUNC(x##y)
#define CAT_FUNC(x, y) __CAT_FUNC(x, y)

#define __CAT_FUNC_CALL(x, y) FUNC_CALL(x##y)
#define CAT_FUNC_CALL(x, y) __CAT_FUNC_CALL(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))


#define ACCUMULATOR_TYPE float
#define UNIT_TYPE float

#define UNIT_VAL_MAX FLT_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0f
#define UNIT_VAL_ZERO 0.0f
#define TO_UNIT_TYPE(v) (float)(v)

#define MAKE_VECTOR_TYPE(elem_type, size) CAT(elem_type, size)




#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_intel_subgroups_short)
#pragma OPENCL EXTENSION  cl_intel_subgroups_short : enable
#endif

#define TRANSPOSE_BLOCK_8( _block )   \
        (float8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_FP16( _block )   \
        (half8)( intel_sub_group_shuffle( _block, 0 ), \
                  intel_sub_group_shuffle( _block, 1 ), \
                  intel_sub_group_shuffle( _block, 2 ), \
                  intel_sub_group_shuffle( _block, 3 ), \
                  intel_sub_group_shuffle( _block, 4 ), \
                  intel_sub_group_shuffle( _block, 5 ), \
                  intel_sub_group_shuffle( _block, 6 ), \
                  intel_sub_group_shuffle( _block, 7 ) );

#define TRANSPOSE_BLOCK_8_COL( _block, _col )   \
        (float8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_8_COL_FP16( _block, _col )   \
        (half8)( intel_sub_group_shuffle( _block.s0, _col ), \
                  intel_sub_group_shuffle( _block.s1, _col ), \
                  intel_sub_group_shuffle( _block.s2, _col ), \
                  intel_sub_group_shuffle( _block.s3, _col ), \
                  intel_sub_group_shuffle( _block.s4, _col ), \
                  intel_sub_group_shuffle( _block.s5, _col ), \
                  intel_sub_group_shuffle( _block.s6, _col ), \
                  intel_sub_group_shuffle( _block.s7, _col ) );

#define TRANSPOSE_BLOCK_16_FP16(_block)  \
        (half16)(as_half2(intel_sub_group_shuffle(_block, 0)),  \
                 as_half2(intel_sub_group_shuffle(_block, 1)),  \
                 as_half2(intel_sub_group_shuffle(_block, 2)),  \
                 as_half2(intel_sub_group_shuffle(_block, 3)),  \
                 as_half2(intel_sub_group_shuffle(_block, 4)),  \
                 as_half2(intel_sub_group_shuffle(_block, 5)),  \
                 as_half2(intel_sub_group_shuffle(_block, 6)),  \
                 as_half2(intel_sub_group_shuffle(_block, 7)));

#define TRANSPOSE_BLOCK_16_FP16_HALF_TYPE(_block)  \
        (half16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

#define TRANSPOSE_BLOCK_16(_block)  \
        (float16)(intel_sub_group_shuffle(_block, 0),  \
                 intel_sub_group_shuffle(_block, 1),  \
                 intel_sub_group_shuffle(_block, 2),  \
                 intel_sub_group_shuffle(_block, 3),  \
                 intel_sub_group_shuffle(_block, 4),  \
                 intel_sub_group_shuffle(_block, 5),  \
                 intel_sub_group_shuffle(_block, 6),  \
                 intel_sub_group_shuffle(_block, 7),  \
                 intel_sub_group_shuffle(_block, 8),  \
                 intel_sub_group_shuffle(_block, 9),  \
                 intel_sub_group_shuffle(_block, 10),  \
                 intel_sub_group_shuffle(_block, 11),  \
                 intel_sub_group_shuffle(_block, 12),  \
                 intel_sub_group_shuffle(_block, 13),  \
                 intel_sub_group_shuffle(_block, 14),  \
                 intel_sub_group_shuffle(_block, 15));

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
{   \
        _result.s0 = mad( _rowA, intel_sub_group_shuffle( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, intel_sub_group_shuffle( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, intel_sub_group_shuffle( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, intel_sub_group_shuffle( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, intel_sub_group_shuffle( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, intel_sub_group_shuffle( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, intel_sub_group_shuffle( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, intel_sub_group_shuffle( colB, 7 ), _result.s7 );  \
}

#define ADD_BIAS_8( _result, _biasVal) \
{ \
    _result.s0 += intel_sub_group_shuffle( _biasVal, 0 ); \
    _result.s1 += intel_sub_group_shuffle( _biasVal, 1 ); \
    _result.s2 += intel_sub_group_shuffle( _biasVal, 2 ); \
    _result.s3 += intel_sub_group_shuffle( _biasVal, 3 ); \
    _result.s4 += intel_sub_group_shuffle( _biasVal, 4 ); \
    _result.s5 += intel_sub_group_shuffle( _biasVal, 5 ); \
    _result.s6 += intel_sub_group_shuffle( _biasVal, 6 ); \
    _result.s7 += intel_sub_group_shuffle( _biasVal, 7 ); \
}

#define ADD_BIAS_16_FP16( _result, _biasVal) \
{ \
    _result.s01 += as_half2(intel_sub_group_shuffle(_biasVal, 0)); \
    _result.s23 += as_half2(intel_sub_group_shuffle(_biasVal, 1)); \
    _result.s45 += as_half2(intel_sub_group_shuffle(_biasVal, 2)); \
    _result.s67 += as_half2(intel_sub_group_shuffle(_biasVal, 3)); \
    _result.s89 += as_half2(intel_sub_group_shuffle(_biasVal, 4)); \
    _result.sab += as_half2(intel_sub_group_shuffle(_biasVal, 5)); \
    _result.scd += as_half2(intel_sub_group_shuffle(_biasVal, 6)); \
    _result.sef += as_half2(intel_sub_group_shuffle(_biasVal, 7)); \
}







#if FP16_UNIT_USED
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))
    #define ALIGNED_ROW_READ8(ptr, element_offset) ( *(const __global half8 *)( ((const __global ushort* *)(ptr) + (element_offset))) )
    
    #define MULTIPLY_BLOCKS_16x8_8x16(_result, _blockA, _blockB) \
    { \
        const half16 acol0 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s0 ); \
        const half16 acol1 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s1 ); \
        const half16 acol2 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s2 ); \
        const half16 acol3 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s3 ); \
        const half16 acol4 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s4 ); \
        const half16 acol5 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s5 ); \
        const half16 acol6 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s6 ); \
        const half16 acol7 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s7 ); \
        _result = fma( _blockB.s0, acol0, _result ); \
        _result = fma( _blockB.s1, acol1, _result ); \
        _result = fma( _blockB.s2, acol2, _result ); \
        _result = fma( _blockB.s3, acol3, _result ); \
        _result = fma( _blockB.s4, acol4, _result ); \
        _result = fma( _blockB.s5, acol5, _result ); \
        _result = fma( _blockB.s6, acol6, _result ); \
        _result = fma( _blockB.s7, acol7, _result ); \
    }
#else
    // Block read - currently block is 4 bytes aligned.
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (byte_offset)))
    #define ALIGNED_ROW_READ8(ptr, element_offset) ( *(const __global float8 *)( ((const __global uint*)(ptr) + (element_offset))) )

    #define MULTIPLY_BLOCKS_16x8_8x16(_result, _blockA, _blockB) \
    { \
        const float16 acol0 = TRANSPOSE_BLOCK_16( _blockA.s0 ); \
        const float16 acol1 = TRANSPOSE_BLOCK_16( _blockA.s1 ); \
        const float16 acol2 = TRANSPOSE_BLOCK_16( _blockA.s2 ); \
        const float16 acol3 = TRANSPOSE_BLOCK_16( _blockA.s3 ); \
        const float16 acol4 = TRANSPOSE_BLOCK_16( _blockA.s4 ); \
        const float16 acol5 = TRANSPOSE_BLOCK_16( _blockA.s5 ); \
        const float16 acol6 = TRANSPOSE_BLOCK_16( _blockA.s6 ); \
        const float16 acol7 = TRANSPOSE_BLOCK_16( _blockA.s7 ); \
        _result = fma( _blockB.s0, acol0, _result ); \
        _result = fma( _blockB.s1, acol1, _result ); \
        _result = fma( _blockB.s2, acol2, _result ); \
        _result = fma( _blockB.s3, acol3, _result ); \
        _result = fma( _blockB.s4, acol4, _result ); \
        _result = fma( _blockB.s5, acol5, _result ); \
        _result = fma( _blockB.s6, acol6, _result ); \
        _result = fma( _blockB.s7, acol7, _result ); \
    }
#endif

typedef float INPUT0_TYPE;
typedef float OUTPUT_TYPE;
typedef float FILTER_TYPE;
typedef float BIAS_TYPE;

#define INPUT0_FEATURE_PITCH 1
#define OUTPUT_FEATURE_PITCH 1

#define GET_DATA_INDEX(O,B,OBASE,Y,X) \
    OBASE + (Y*OUTPUT_SIZE_X + X) * INPUT0_FEATURE_NUM 


__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Conv2D_1x1(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
    __global BIAS_TYPE* biases )
{
   // This
   // group[0] = output pixels strip of sixteen (W*H) / 16
   // group[1] = output features / 16

    const uint threadId =  get_sub_group_local_id();
    const uint xyBase = get_group_id(0) * 16;
    const uint xy_t = xyBase + threadId;
    const uint inputPixel_t = xy_t * INPUT0_FEATURE_NUM;
    const uint x_t = xy_t % OUTPUT_SIZE_X;
    const uint y_t = xy_t / OUTPUT_SIZE_X;
    const uint baseOutputFeature = get_group_id(1) * 16;
    const uint outputFeature_t = baseOutputFeature + get_sub_group_local_id();


    // 16 consecutive outputs for out xy...
    float16  output16s;


    // Each thread loads 1 bias...
    float threadBiasValue = biases[threadId];

    // Then these are shared to other threads
    for(uint i = 0; i < 16; i++)
    {
        output16s[i] = intel_sub_group_shuffle(threadBiasValue, i);
    }

    const uint dst_base = xy_t * OUTPUT_FEATURE_NUM + baseOutputFeature;

    for (uint k = 0; k < (INPUT0_FEATURE_NUM/8) ; ++k)
    {
        // Load 8 inputs from 16 pixels
        uint input_idx = inputPixel_t + k * 8;

        // TODO - reading past end of array - need to pad to up to 16 pixels at end of buffer


        // this will read pixels: base,  base + 16, base + 32, ... base + 16*7
        //  not what I want, although it seems slightly faster than the ROW method
        //float8 inputs8x16 = ALIGNED_BLOCK_READ8(input, input_idx);
        float8 inputs8x16 = ALIGNED_ROW_READ8(input, input_idx);

        uint weight_idx = outputFeature_t * INPUT0_FEATURE_NUM + k*8;

        // Load 8 input-weights for 16 outputs
        //float8 weights8x16  = ALIGNED_BLOCK_READ8(weights, weight_idx); // Read 8 weights, 16 threads
        float8 weights8x16  = ALIGNED_ROW_READ8(weights, weight_idx); // Read 8 weights, 16 threads

        // MAD 16 lots of 8 pixels, 8 weights  and accumulate into 16
        MULTIPLY_BLOCKS_16x8_8x16(output16s, weights8x16, inputs8x16);
    }

    // Don't write past end of array
    if( xy_t >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;

    // Write output to this threads pixel
    for(uint i = 0; i < 16; i++)
    {
        const uint dst_index = dst_base + i;
    //#if LEFTOVERS
        //if(group_f+i < OUTPUT_FEATURE_NUM)
    //#endif
        output[dst_index] = ACTIVATION(output16s[i]);
    }
}

