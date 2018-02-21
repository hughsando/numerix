

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

#define READ_T8(ptr,element)  as_float(intel_sub_group_block_read((const __global uint*)(ptr + element)))


__attribute__((reqd_work_group_size(1,1,8)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Deconv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *bias)
{
    const int tileId = get_global_id(0);
    const int outBase = get_global_id(1)*8;
    const int threadId = get_local_id(2);
    const int srcFx0 = (tileId % SRC_TX) + ( (-PAD_X)<<SHIFT_X );
    const int srcFy0 = (tileId / SRC_TX) + ( (-PAD_Y)<<SHIFT_Y );

    const int destY0 = (srcFy0<<SHIFT_Y) + PAD_Y;
    const int destX0 = (srcFx0<<SHIFT_X) + PAD_X;

    // Tile output,
    float output_t8[TH][TW]/* x8 theads */;

    // Source tile
    float src8[SH][SW] /* x8 threads */;

    const __global float *w0 = weights + outBase*INPUTS*FILTER_X*FILTER_Y;

    // Init with bias...
    output_t8[0][0] = READ_T8(bias,outBase);
    // Copy to outher outputs
    UNROLL_OUTPUT(I, UNROLL_OUTPUT(J, { if (I||J) output_t8[I][J] = output_t8[0][0]; } ) );

    for(int i=0;i<INPUTS;i+=8)
    {
       const __global float *srcI = src + i;
       UNROLL_INPUT(Y,{
         int sy=Y+srcFy0;
         bool validY = (sy>=0 && sy<SRC_H);
         UNROLL_INPUT(X,{
               int sx=X+srcFx0;
               src8[Y][X] = ( validY && (sx>=0 && sx<SRC_W) ) ?  READ_T8(srcI,(sy*SRC_W + sx)*INPUTS ) : 0.0f;
            })
         })

       UNROLL_OUTPUT(Y,{
          int y = destY0 + Y;
          UNROLL_OUTPUT(X,{
             int x = destX0 + X;

             const __global float8 *w = ((const __global float8 *)(w0 + 8*SW*SH*8*( Y*TW + X) )) + threadId;

             UNROLL_INPUT(SY,{
                UNROLL_INPUT(SX,{
                       float8 w8 = w[ (SY*SW+SX)*8 ];

                       #define ACCUMULATE( output8, src8, w8 ) \
                           output8 = mad( sub_group_broadcast(src8,0), w8.s0, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,1), w8.s1, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,2), w8.s2, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,3), w8.s3, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,4), w8.s4, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,5), w8.s5, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,6), w8.s6, output8 ); \
                           output8 = mad( sub_group_broadcast(src8,7), w8.s7, output8 ); \


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
                d0[ (y*DEST_W+x)*OUTPUTS ] = ACTIVATION( output_t8[Y][X] );
             }
          })
       }
    })
};



