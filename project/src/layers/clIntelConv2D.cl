
#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_intel_subgroups_short)
#pragma OPENCL EXTENSION  cl_intel_subgroups_short : enable
#endif


typedef float INPUT0_TYPE;
typedef float OUTPUT_TYPE;
typedef float FILTER_TYPE;
typedef float BIAS_TYPE;

#define SRC_W INPUT0_SIZE_X
#define SRC_H INPUT0_SIZE_Y
#define DEST_W OUTPUT_SIZE_X
#define DEST_H OUTPUT_SIZE_Y

typedef const __global float8 *f8Ptr;
typedef const __global float4 *f4Ptr;
typedef const __global float2 *f2Ptr;

#define READ_T8(ptr,element)  as_float(intel_sub_group_block_read((const __global uint*)(ptr + element)))
#define READ_T8x8(ptr,element)  as_float8(intel_sub_group_block_read8((const __global uint*)(ptr + element)))


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

#ifdef INTEL_TILED_1X1

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

    const bool validPixel_t = xy_t < DEST_W*DEST_H;


    // Each thread loads 1 bias...
    float threadBiasValue = biases[baseOutputFeature + threadId];

    // Then these are shared to other threads
    for(uint i = 0; i < 16; i++)
    {
        output16s[i] = intel_sub_group_shuffle(threadBiasValue, i);
    }


    for (uint k = 0; k < (INPUT0_FEATURE_NUM/8) ; ++k)
    {
        // Load 8 inputs from 16 pixels
        uint input_idx_t = inputPixel_t + k * 8;

        // this will read pixels: base,  base + 16, base + 32, ... base + 16*7
        //  not what I want, although it seems slightly faster than the ROW method
        //float8 inputs8x16 = ALIGNED_BLOCK_READ8(input, input_idx);
        float8 inputs8x16 = validPixel_t ? ( ALIGNED_ROW_READ8(input, input_idx_t) ) : 0.0f;

        uint weight_idx = outputFeature_t * INPUT0_FEATURE_NUM + k*8;

        // Load 8 input-weights for 16 outputs
        //float8 weights8x16  = ALIGNED_BLOCK_READ8(weights, weight_idx); // Read 8 weights, 16 threads
        float8 weights8x16  = ALIGNED_ROW_READ8(weights, weight_idx); // Read 8 weights, 16 threads

        barrier(CLK_LOCAL_MEM_FENCE);

        // MAD 16 lots of 8 pixels, 8 weights  and accumulate into 16
        MULTIPLY_BLOCKS_16x8_8x16(output16s, weights8x16, inputs8x16);
    }

    // Don't write past end of array
    if (validPixel_t)
    {
      const uint dst_base = xy_t * OUTPUT_FEATURE_NUM + baseOutputFeature;
      // Write output to this threads pixel
      for(uint i = 0; i < 16; i++)
      {
          output[dst_base+i] = ACTIVATION(output16s[i]);
      }
    }
}

#endif // INTEL_TILED_1X1




#ifdef INTEL_TILED_3X3

// Convolution for 3x3 filter, 6xTY tiles
// Each workgroup calculates 8 outputs, and has 8 threads

// Tile Height
#define TW 4
#define TH 4




__attribute__((reqd_work_group_size(1, 1, 8)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Conv2D_3x3(
    const __global float *input,
    __global float *dest,
    const __global float *weights,
    const __global float *bias )
{
    const unsigned xTile = get_group_id(0);
    const unsigned yTile = get_group_id(1);
    const unsigned outBase = get_group_id(2) * 8;

    // threadId is used for input and output feature index
    const unsigned threadId = get_local_id(2);

    const int x0 = xTile * TW;
    const int y0 = yTile * TH;
    const __global float *srcPtr0 = input + ( (y0-1)*SRC_W + (x0-1))*INPUT0_FEATURE_NUM;

    // Tile output,
    float output_t8[TH][TW]/* x8 theads */;
    // Source rows
    float srcY_t8[3][TW+2] /* x8 threads */;


    #define UNROLL_TILE(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
       { const int VAR =2; loop; } \
       { const int VAR =3; loop; } \
    }
    #define UNROLL_OUTER(VAR,loop) { \
       UNROLL_TILE(VAR,loop) \
       { const int VAR =4; loop; } \
       { const int VAR =5; loop; } \
    }


    // Init with bias...
    output_t8[0][0] = READ_T8(bias,outBase);
    // Copy to outher outputs
    UNROLL_TILE(I, UNROLL_TILE(J, { if (I||J) output_t8[I][J] = output_t8[0][0]; } ) );


    // output8 is the 8 feature outputs at a given location
    //    float * thread8
    // src8 is the 8 feature src inputs at convolution offset
    //    float * thread8
    // w8 is the 8 weights that must be applied to each src to accumulate in the output
    //    is float8 * thread8
    //     =  [INPUT] * [ OUTPUT ]
    //
    // fma vs mad?
    #define ACCUMULATE( output8, src8, w8 ) \
        output8 = mad( sub_group_broadcast(src8,0), w8.s0, output8 ); \
        output8 = mad( sub_group_broadcast(src8,1), w8.s1, output8 ); \
        output8 = mad( sub_group_broadcast(src8,2), w8.s2, output8 ); \
        output8 = mad( sub_group_broadcast(src8,3), w8.s3, output8 ); \
        output8 = mad( sub_group_broadcast(src8,4), w8.s4, output8 ); \
        output8 = mad( sub_group_broadcast(src8,5), w8.s5, output8 ); \
        output8 = mad( sub_group_broadcast(src8,6), w8.s6, output8 ); \
        output8 = mad( sub_group_broadcast(src8,7), w8.s7, output8 ); \

    int weightBase = 9*outBase*INPUT0_FEATURE_NUM;

    // Loads up [XY=9][INPUTS=float8][OUTPUTS=thread]
    float8 w_t8[9] /* x8 */;

    for(int i=0; i<INPUT0_FEATURE_NUM; i+=8)
    {
       w_t8[0] = READ_T8x8(weights, weightBase+8*8*0);
       w_t8[1] = READ_T8x8(weights, weightBase+8*8*1);
       w_t8[2] = READ_T8x8(weights, weightBase+8*8*2);
       w_t8[3] = READ_T8x8(weights, weightBase+8*8*3);
       w_t8[4] = READ_T8x8(weights, weightBase+8*8*4);
       w_t8[5] = READ_T8x8(weights, weightBase+8*8*5);
       w_t8[6] = READ_T8x8(weights, weightBase+8*8*6);
       w_t8[7] = READ_T8x8(weights, weightBase+8*8*7);
       w_t8[8] = READ_T8x8(weights, weightBase+8*8*8);


       weightBase += 9*8*8;

       // Pre-load first 2 rows ...
       UNROLL_OUTER(X, {
          int x = x0+(X-1);
          srcY_t8[0][X] = x>=0 && x<SRC_W && y0>0 ? READ_T8( srcPtr0, X*INPUT0_FEATURE_NUM ) : 0.0f;
          srcY_t8[1][X] = x>=0 && x<SRC_W ?         READ_T8( srcPtr0, (SRC_W+X)*INPUT0_FEATURE_NUM ) : 0.0f;
       } );

       UNROLL_TILE(Y, {
          const int s0 = Y%3;
          const int s1 = (Y+1)%3;
          const int s2 = (Y+2)%3;
          if (y0+Y<DEST_H)
          {
             UNROLL_TILE(X, {
                ACCUMULATE( output_t8[Y][X], srcY_t8[s0][1+X-1], w_t8[0] );
                ACCUMULATE( output_t8[Y][X], srcY_t8[s0][1+X  ], w_t8[1] );
                ACCUMULATE( output_t8[Y][X], srcY_t8[s0][1+X+1], w_t8[2] );

                ACCUMULATE( output_t8[Y][X], srcY_t8[s1][1+X-1], w_t8[3] );
                ACCUMULATE( output_t8[Y][X], srcY_t8[s1][1+X  ], w_t8[4] );
                ACCUMULATE( output_t8[Y][X], srcY_t8[s1][1+X+1], w_t8[5] );
             })

             if (y0+Y+1<DEST_H )
             {
                // Load bottom row
                UNROLL_OUTER(X, {
                   int x = x0+(X-1);
                   srcY_t8[s2][X] = x>=0 && x<SRC_W ? READ_T8( srcPtr0, (SRC_W*(Y+2)+X)*INPUT0_FEATURE_NUM ) : 0.0f;
                } );
                UNROLL_TILE(X, {
                   ACCUMULATE( output_t8[Y][X], srcY_t8[s2][1+X-1], w_t8[6] );
                   ACCUMULATE( output_t8[Y][X], srcY_t8[s2][1+X  ], w_t8[7] );
                   ACCUMULATE( output_t8[Y][X], srcY_t8[s2][1+X+1], w_t8[8] );
                })
             }
          }
       } );

       srcPtr0 += 8;
    }

    UNROLL_TILE(Y, {
       UNROLL_TILE(X, {
          float o = ACTIVATION( output_t8[Y][X] );
          //dest[  ((y0+Y)*DEST_W+(x0+X))*OUTPUT_FEATURE_NUM + outBase + threadId ] =  o;
          intel_sub_group_block_write((__global uint*)( dest + ( (y0+Y)*DEST_W+(x0+X))*OUTPUT_FEATURE_NUM + outBase) , as_uint(o) );
       })
    });
}

#endif






#ifdef INTEL_TILED_3X3X3

// Convolution for 3x3 filter, 6xTY tiles
// Each workgroup calculates 8 outputs, and has 8 threads

// Output Tile Height
#define OTW 3
#define OTH 4

// stride = 2



__attribute__((reqd_work_group_size(1, 1, 8)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Conv2D_3X3X3(
    const __global float *input,
    __global float *dest,
    const __global float *weights,
    const __global float *bias )
{
    const unsigned xTile = get_group_id(0);
    const unsigned yTile = get_group_id(1);
    const unsigned outBase = get_group_id(2) * 8;

    // threadId is used for source columnn and output feature index
    const signed int threadId = get_local_id(2);

    const signed int sx0 = xTile * OTW * 2 - PADX;
    const signed int sy0 = yTile * OTH * 2 - PADY;


    // 4x3 x 3 -> 9 x 7 x 3
    // Source rows
    float src[9][3]; // x t8 - only need 7 compnents, so last thread can slack off


    #define UNROLL_SRC_ROWS(VAR,loop) { \
       { const signed int VAR =0; loop; } \
       { const signed int VAR =1; loop; } \
       { const signed int VAR =2; loop; } \
       { const signed int VAR =3; loop; } \
       { const signed int VAR =4; loop; } \
       { const signed int VAR =5; loop; } \
       { const signed int VAR =6; loop; } \
       { const signed int VAR =7; loop; } \
       { const signed int VAR =8; loop; } \
    }

    int  srcOffset = (sy0*SRC_W + sx0 + threadId)*INPUT0_FEATURE_NUM;
    // Only load valid x, and if threadId is ok ...
    bool validX = (sx0+threadId>=0) &&  (threadId < 7) && (sx0+threadId<SRC_W );
    UNROLL_SRC_ROWS(Y, {
          bool valid = validX && (sy0+Y>=0) && (sy0+Y < SRC_H);
          src[Y][0] = valid ? input[ srcOffset ]   : 0.0f;
          src[Y][1] = valid ? input[ srcOffset + 1]: 0.0f;
          src[Y][2] = valid ? input[ srcOffset + 2]: 0.0f;
          srcOffset += SRC_W * INPUT0_FEATURE_NUM;
       });

    #define UNROLL_WIDTH(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
       { const int VAR =2; loop; } \
    }
    #define UNROLL_HEIGHT(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
       { const int VAR =2; loop; } \
       { const int VAR =3; loop; } \
    }


    const int dx0 = xTile * OTW;
    const int dy0 = yTile * OTH;

    int weightBase = (outBase + threadId) * 3 * 3 *3;

    // Load weights
    union {
       float w_t8[9][3];
       struct {
          float8 w8[3];
          float2 w2;
          float  w1;
       } s;
    } w;
    // TODO - reorder weights
    // For each output, load 9x3 (27) 
    w.s.w8[0] = *(f8Ptr)( weights + weightBase );
    w.s.w8[1] = *(f8Ptr)( weights + weightBase + 8);
    w.s.w8[2] = *(f8Ptr)( weights + weightBase + 16);
    w.s.w2 =    *(f2Ptr)( weights + weightBase + 24);
    w.s.w1 =    *       ( weights + weightBase + 26);

    barrier(CLK_LOCAL_MEM_FENCE);

   // Accumulate out
   #define ACCUMULATE( output8, src, x, w ) \
        output8 = mad( intel_sub_group_shuffle(src[0],x), w[0], output8 ); \
        output8 = mad( intel_sub_group_shuffle(src[1],x), w[1], output8 ); \
        output8 = mad( intel_sub_group_shuffle(src[2],x), w[2], output8 ); \


    float biasVal = READ_T8(bias,outBase);

    UNROLL_HEIGHT(Y, {
          int oy = dy0+Y;
          UNROLL_WIDTH(X, {
             int ox = dx0 + X;
             if ( (ox<DEST_W) && (oy<DEST_H) )
             {
                float sum = biasVal;


                ACCUMULATE( sum, src[Y*2+0],X*2   , w.w_t8[0] );
                ACCUMULATE( sum, src[Y*2+0],X*2+1 , w.w_t8[1] );
                ACCUMULATE( sum, src[Y*2+0],X*2+2 , w.w_t8[2] );

                ACCUMULATE( sum, src[Y*2+1],X*2  , w.w_t8[3] );
                ACCUMULATE( sum, src[Y*2+1],X*2+1, w.w_t8[4] );
                ACCUMULATE( sum, src[Y*2+1],X*2+2, w.w_t8[5] );

                ACCUMULATE( sum, src[Y*2+2],X*2  , w.w_t8[6] );
                ACCUMULATE( sum, src[Y*2+2],X*2+1, w.w_t8[7] );
                ACCUMULATE( sum, src[Y*2+2],X*2+2, w.w_t8[8] );

                float o = ACTIVATION(sum);

                //dest[( oy*DEST_W+ox)*OUTPUT_FEATURE_NUM + outBase + threadId] = o;
                intel_sub_group_block_write((__global uint*)( dest + ( oy*DEST_W+ox)*OUTPUT_FEATURE_NUM + outBase) , as_uint(o) );
             }
          })
    } );
}

#endif



