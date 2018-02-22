

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#define TW (1<<SHIFT_X)
#define TH (1<<SHIFT_Y)

#define SW (FILTER_X>>SHIFT_X)
#define SH (FILTER_Y>>SHIFT_Y)


#if (TH==4)
    #define UNROLL_OUTPUT(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
       { const int VAR =2; loop; } \
       { const int VAR =3; loop; } \
    }
#elif (TH==2)
    #define UNROLL_OUTPUT(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
    }
#else
    #error "Unsupported output size"
#endif


#if (SH==4)
    #define UNROLL_INPUT(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
       { const int VAR =2; loop; } \
       { const int VAR =3; loop; } \
    }
#elif (SH==2)
    #define UNROLL_INPUT(VAR,loop) { \
       { const int VAR =0; loop; } \
       { const int VAR =1; loop; } \
    }
#elif (SH==1)
    #define UNROLL_INPUT(VAR,loop) { \
       { const int VAR =0; loop; } \
    }
#else
    #error "Unsupported input size"
#endif

#define WEIGHT_SIZE ( SW*SH*INPUTS )


#ifdef INTEL

   #define FLOAT_T8( name ) float name
   #define READ_T8( ptr,element) as_float(intel_sub_group_block_read((const __global uint*)(ptr + element)))
   #define REF_T8(name) name
   #define READ_SYNC

   #define ACCUMULATE( output8, src8, w8 ) \
      output8 = mad( sub_group_broadcast(src8,0), w8.s0, output8 ); \
      output8 = mad( sub_group_broadcast(src8,1), w8.s1, output8 ); \
      output8 = mad( sub_group_broadcast(src8,2), w8.s2, output8 ); \
      output8 = mad( sub_group_broadcast(src8,3), w8.s3, output8 ); \
      output8 = mad( sub_group_broadcast(src8,4), w8.s4, output8 ); \
      output8 = mad( sub_group_broadcast(src8,5), w8.s5, output8 ); \
      output8 = mad( sub_group_broadcast(src8,6), w8.s6, output8 ); \
      output8 = mad( sub_group_broadcast(src8,7), w8.s7, output8 ); \


#else

   #define FLOAT_T8( name ) __local float name[8]
   #define READ_T8(ptr,element) ((const __global float*)(ptr + element))[threadId]
   #define REF_T8(name) name[threadId]
   #define READ_SYNC  barrier(CLK_LOCAL_MEM_FENCE)

   #define ACCUMULATE( output8, src8, w8 ) \
      output8[threadId] = mad( src8[0], w8.s0, output8[threadId] ); \
      output8[threadId] = mad( src8[1], w8.s1, output8[threadId] ); \
      output8[threadId] = mad( src8[2], w8.s2, output8[threadId] ); \
      output8[threadId] = mad( src8[3], w8.s3, output8[threadId] ); \
      output8[threadId] = mad( src8[4], w8.s4, output8[threadId] ); \
      output8[threadId] = mad( src8[5], w8.s5, output8[threadId] ); \
      output8[threadId] = mad( src8[6], w8.s6, output8[threadId] ); \
      output8[threadId] = mad( src8[7], w8.s7, output8[threadId] ); \



#endif


#ifndef ODD_DECONV2D

__attribute__((reqd_work_group_size(1,1,8)))
#ifdef INTEL
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Deconv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *bias)
{
    const int tileId = get_global_id(0);
    const int outBase = get_global_id(1)*8;
    const int threadId = get_local_id(2);

    const int srcFx0 = (tileId % SRC_TX) + (((PAD_X) >> SHIFT_X) - SW/2);
    const int srcFy0 = (tileId / SRC_TX) + (((PAD_Y) >> SHIFT_Y) - SH/2);

    const int destY0 = (( srcFy0 + SH/2)<<SHIFT_Y) - PAD_Y;
    const int destX0 = (( srcFx0 + SW/2)<<SHIFT_X) - PAD_X;

    // Tile output,
    FLOAT_T8( output_t8[TH][TW] ) /* x8 theads */;

    // Source tile
    FLOAT_T8( src8[SH][SW] ) /* x8 threads */;

    const __global float *w0 = weights + outBase*INPUTS*FILTER_X*FILTER_Y;

    // Init with bias...
    REF_T8( output_t8[0][0] ) = READ_T8(bias,outBase);

    // Copy to outher outputs
    UNROLL_OUTPUT(I, UNROLL_OUTPUT(J, { if (I||J) REF_T8(output_t8[I][J]) = REF_T8(output_t8[0][0]); } ) );

    for(int i=0;i<INPUTS;i+=8)
    {
       const __global float *srcI = src + i;
       UNROLL_INPUT(Y,{
         int sy=Y+srcFy0;
         bool validY = (sy>=0 && sy<SRC_H);
         UNROLL_INPUT(X,{
               int sx=X+srcFx0;
               REF_T8(src8[Y][X]) = ( validY && (sx>=0 && sx<SRC_W) ) ?  READ_T8(srcI,(sy*SRC_W + sx)*INPUTS ) : 0.0f;
            })
         })

       READ_SYNC;

       UNROLL_OUTPUT(Y,{
          int y = destY0 + Y;
          UNROLL_OUTPUT(X,{
             int x = destX0 + X;

             const __global float8 *w = ((const __global float8 *)(w0 + 8*SW*SH*8*( Y*TW + X) )) + threadId;

             UNROLL_INPUT(SY,{
                UNROLL_INPUT(SX,{
                       float8 w8 = w[ (SY*SW+SX)*8 ];

                       // One output for 8 inputs
                       ACCUMULATE(output_t8[Y][X], src8[SY][SX], w8)
                   })
             })
          })
       });

       w0 += (SW*SH*8*8)<<(SHIFT_X + SHIFT_Y);
    }

    __global float *d0 = dest + outBase + threadId;
    UNROLL_OUTPUT(Y,{
       int y = destY0 + Y;
       if (y>=0 && y<=DEST_H)
       {
          UNROLL_OUTPUT(X,{
             int x = destX0 + X;
             if (x>=0 && x<DEST_W)
             {
                d0[ (y*DEST_W+x)*OUTPUTS ] = ACTIVATION( REF_T8( output_t8[Y][X] ) );
             }
          })
       }
    })
};

#else
// Odd - not much data-parallelism




__kernel void Deconv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *bias)
{
    const int tileId = get_global_id(0);
    const int srcFx0 = (tileId % SRC_TX) + (((PAD_X) >> SHIFT_X) - SW/2);
    const int srcFy0 = (tileId / SRC_TX) + (((PAD_Y) >> SHIFT_Y) - SH/2);

    const int destY0 = (( srcFy0 + SH/2)<<SHIFT_Y) - PAD_Y;
    const int destX0 = (( srcFx0 + SW/2)<<SHIFT_X) - PAD_X;

    // Source tile
    float srcTile[SH][SW][INPUTS];

    //const __global float *w0 = weights + outBase*INPUTS*FILTER_X*FILTER_Y;

    UNROLL_INPUT(Y,{
      int sy=Y+srcFy0;
      bool validY = (sy>=0 && sy<SRC_H);
      UNROLL_INPUT(X,{
            int sx=X+srcFx0;
            if ( validY && (sx>=0 && sx<SRC_W) )
            {
               const __global float *srcI = src + (sy*SRC_W + sx)*INPUTS;
               for(int i=0;i<INPUTS;i++)
                  srcTile[Y][X][i] = srcI[i];
            }
            else
            {
               for(int i=0;i<INPUTS;i++)
                  srcTile[Y][X][i] = 0.0f;
            }
         })
      })


   const __global float *w = weights;

    UNROLL_OUTPUT(Y,{
       int y = destY0 + Y;
       UNROLL_OUTPUT(X,{
          int x = destX0 + X;

          for(int o=0;o<OUTPUTS;o++)
          {
             float sum = bias[o];

             for(int i=0;i<INPUTS;i++)
                UNROLL_INPUT(SY,{
                   UNROLL_INPUT(SX,{
                      sum += *w++ * srcTile[SY][SX][i];
                   })
                })
             if (y>=0 && y<=DEST_H && x>=0 && x<DEST_W)
                dest[ (y*DEST_W+x)*OUTPUTS + o] = ACTIVATION( sum );
          }

       })
    });
};










#endif



