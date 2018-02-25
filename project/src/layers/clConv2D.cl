



#ifdef CONV2D_O8I8

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#define SW (FILTER_X+TW-1)
#define SH (FILTER_Y+TH-1)


#define LOOP(VAR,N,loop) {\
    if (N>0) { const int VAR =0; loop; } \
    if (N>1) { const int VAR =1; loop; } \
    if (N>2) { const int VAR =2; loop; } \
    if (N>3) { const int VAR =3; loop; } \
    if (N>4) { const int VAR =4; loop; } \
    if (N>5) { const int VAR =5; loop; } \
    if (N>6) { const int VAR =6; loop; } \
    if (N>7) { const int VAR =7; loop; } \
    if (N>8) { const int VAR =8; loop; } \
    if (N>9) { const int VAR =9; loop; } \
}

#define WEIGHT_SIZE ( FILTER_X*FILTER_Y*INPUTS )


#ifdef INTEL_SUBGROUPS

   #define FLOAT_T8( name ) float name
   #define FLOAT_T8x8( name ) float8 name
   #define READ_T8( ptr,element) as_float(intel_sub_group_block_read((const __global uint*)(ptr + element)))
   #define REF_T8(name) name
   #define READ_SYNC

   #define READ_T8_Thread(ptr,element)  as_float8(intel_sub_group_block_read8((const __global uint*)(ptr + element)))

   #define ACCUMULATEx8( output8, src8, w8 ) \
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
   #define FLOAT_T8x8( name ) __local float8 name[8]
   #define READ_T8(ptr,element) ((const __global float*)(ptr + element))[threadId]
   #define READ_T8_Thread(ptr,element) ((const __global float8*)(ptr + element))[threadId]
   #define REF_T8(name) name[threadId]
   #define READ_SYNC  barrier(CLK_LOCAL_MEM_FENCE)




   #define ACCUMULATEx8( output8, src8, w8 ) \
      output8 = mad( src8[0], w8.s0, output8 ); \
      output8 = mad( src8[1], w8.s1, output8 ); \
      output8 = mad( src8[2], w8.s2, output8 ); \
      output8 = mad( src8[3], w8.s3, output8 ); \
      output8 = mad( src8[4], w8.s4, output8 ); \
      output8 = mad( src8[5], w8.s5, output8 ); \
      output8 = mad( src8[6], w8.s6, output8 ); \
      output8 = mad( src8[7], w8.s7, output8 ); \



#endif



__attribute__((reqd_work_group_size(1,1,8)))
#ifdef INTEL_SUBGROUPS
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Conv2Do8i8(const __global float* src, __global float *dest,
       const __global float *weights, const __global float *bias,
       const int srcW, const int srcH, const int inputs,
       const int destW, const int destH, const int outputs)
{
    const int tileX = get_global_id(0);
    const int tileY = get_global_id(1);
    const int outBase = get_global_id(2)&~7;
    const int threadId = get_local_id(2);

    const int srcFx0 = (tileX * TW) - PAD_X;
    const int srcFy0 = (tileY * TH) - PAD_Y;

    const int destX0 = (tileX * TW);
    const int destY0 = (tileY * TH);

    // Tile output,
    float output_t8[TH][TW] /* x8 theads */;


    // Init with bias...
    output_t8[0][0] = READ_T8(bias,outBase);

    // Copy to outher outputs
    LOOP(J, TH, LOOP(I, TW, { if (I||J) output_t8[J][I] = output_t8[0][0]; } ) );

    for(int i=0;i<inputs;i+=8)
    {
       // Source tile
       FLOAT_T8( src8[SH][SW] ) /* x8 threads */;


       // Load src for this input
       const __global float *srcI = src + i;
       LOOP(Y,SH,{
         int sy=Y+srcFy0;
         bool validY = (sy>=0 && sy<srcH);
         LOOP(X,SW,{
               int sx=X+srcFx0;
               REF_T8(src8[Y][X]) = ( validY && (sx>=0 && sx<srcW) ) ?  READ_T8(srcI,(sy*srcW + sx)*inputs ) : 0.0f;
            })
         })

       READ_SYNC;

       const __global float8 *w0 = (const __global float8 *)(weights + outBase*inputs*FILTER_X*FILTER_Y + i*FILTER_X*FILTER_Y*8);

       LOOP(SY,FILTER_Y,{
           LOOP(SX,FILTER_X,{
              float8 w8 = READ_T8_Thread( w0, ((SY*FILTER_X+SX)*8) );

               LOOP(Y,TH,{
                    LOOP(X,TW,{
                       // 8 output(threads/[threadId]) for 8 inputs(float8 vector)
                       ACCUMULATEx8(output_t8[Y][X], src8[Y+SY][X+SX], w8)
                   })
               })
          })
       });
    }
    __global float *d0 = dest + outBase + threadId;
    LOOP(Y,TH,{
       int y = destY0 + Y;
       if (y>=0 && y<=destH)
       {
          LOOP(X,TW,{
             int x = destX0 + X;
             if (x>=0 && x<destW)
             {
                d0[ (y*destW+x)*outputs ] = ACTIVATION( output_t8[Y][X] );
             }
          })
       }
    })
};




#endif

#define INPUTS4 (INPUTS/4)
#define THREADS OVEC

#ifdef TILED_1X1

inline void processTiles1x1(const int MaxPix,
   const __global float* s1,
   const __global float *weights,
   const __global float *bias,
   __global float *out)
{
   float4 zero = 0.0f;

   // tid = threadId (0...OVEC) used to do 2xOVEC outputs at the same time
   const int tid = get_local_id(0);


   // float4 seem to be fast that float8 on hardware
   for(int o=0;o<OUTPUTS;o+= THREADS*2 )
   {
      // Each thread handles 6 horizontal outputs for 1 output
      float4 sum00 = { bias[o+tid*2], 0.0f, 0.0f, 0.0f };
      float4 sum01 = sum00;
      float4 sum02 = sum00;
      float4 sum03 = sum00;
      float4 sum04 = sum00;
      float4 sum05 = sum00;
      float4 sum10 = { bias[o+tid*2+1], 0.0f, 0.0f, 0.0f };
      float4 sum11 = sum10;
      float4 sum12 = sum10;
      float4 sum13 = sum10;
      float4 sum14 = sum10;
      float4 sum15 = sum10;

      __global const float4 *s4 = (__global const float4 *)(s1);

      // Each thread is running a different output
      const __global float4 *w4 = (const __global float4 *)( weights + (o+tid*2)*INPUTS );

      for(int sx=0; sx<INPUTS4 ; sx++, s4++)
      {
         #define GetSrcOffset(Pixel) (Pixel<MaxPix) ?  s4[ Pixel*INPUTS4 ] : zero;

         float4 weight0 = w4[0];
         float4 weight1 = w4[INPUTS4];
         w4++;

         float4 s = s4[0];
         sum00 = mad(weight0, s, sum00 );
         sum10 = mad(weight1, s, sum10 );
         if (MaxPix==1) continue;

         s = s4[INPUTS4];
         sum01 = mad(weight0, s, sum01 );
         sum11 = mad(weight1, s, sum11 );
         if (MaxPix==2) continue;

         s = s4[INPUTS4*2];
         sum02 = mad(weight0, s, sum02 );
         sum12 = mad(weight1, s, sum12 );
         if (MaxPix==3) continue;

         s = s4[INPUTS4*3];
         sum03 = mad(weight0, s, sum03 );
         sum13 = mad(weight1, s, sum13 );
         if (MaxPix==4) continue;

         s = s4[INPUTS4*4];
         sum04 = mad(weight0, s, sum04 );
         sum14 = mad(weight1, s, sum14 );
         if (MaxPix==5) continue;

         s = s4[INPUTS4*5];
         sum05 = mad(weight0, s, sum05 );
         sum15 = mad(weight1, s, sum15 );
      }


      // Write  output for 6 pixels per thread + bias + activation

      float2 f2;
      float val;
      #define SAVE(X,Y) \
         if (X<MaxPix) { \
         f2 = sum##Y##X.xy + sum##Y##X.zw; \
         val = f2.x + f2.y; \
         out[(tid*2)+Y+OUTPUTS*X] = ACTIVATION(val); \
         }

      SAVE(0,0)
      SAVE(0,1)
      SAVE(1,0)
      SAVE(1,1)
      SAVE(2,0)
      SAVE(2,1)
      SAVE(3,0)
      SAVE(3,1)
      SAVE(4,0)
      SAVE(4,1)
      SAVE(5,0)
      SAVE(5,1)

      out += THREADS*2;
   }
}


__kernel void Conv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *bias, const int width, const int height)
{
   const int tid = get_local_id(0);
   const int row = get_global_id(1);

   const __global float *s1 =  src + (row)*INPUTS*width;

   __global float *out = dest + row*OUTPUTS*width;

   int x = 0;
   for(  ;x+6<=width; x+=6 )
   {
      processTiles1x1(1000, s1, weights, bias, out);

      out += OUTPUTS*6;
      s1+=INPUTS*6;
   }

   if (x<width)
      processTiles1x1(width-x, s1, weights, bias, out);
}


#endif

#ifdef CONV2D_1x1

__kernel void Conv2D(const __global float* inSrc, __global float *dest, const __global float *inWeights, const __global float *inBias, const int width, const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int srcStride = width*INPUTS;
    const int destStride = width*OUTPUTS;
    
    int woff = 0;
    int destOff = y*destStride + OUTPUTS*x;
    int srcOff = y*srcStride + INPUTS*x;
    for(int o=0;o<OUTPUTS;o++) {
       float sum=inBias[o];
       for(int i=0;i<INPUTS;i++)
           sum += inSrc[srcOff+i] * inWeights[woff+i];
       woff+=INPUTS;
    dest[destOff+o] = ACTIVATION(sum);
    }
}

#endif


#ifdef CONV2D_SIMPLE


__kernel void Conv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *inBias, const int srcWidth, const int srcHeight )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int srcStride = srcWidth*INPUTS;

    int destOff = (y*DEST_W+x) * OUTPUTS;
    int weightOff = 0;
    for(int o=0;o<OUTPUTS;o++)
    {
       float sum=inBias[o];
       for(int dy=dMin; dy<dMax; dy++)
       {
          int sy = y*STRIDE_Y+dy;
          if (sy>=0 && sy<srcHeight)
          {
             for(int dx=dMin; dx<dMax; dx++)
             {
                 int sx = x*STRIDE_X+dx;
                 if (sx>=0 && sx<srcWidth)
                 {
                    int srcOff = sy*srcStride + INPUTS*sx;
                    for(int i=0;i<INPUTS;i++)
                        sum += src[srcOff+i] * weights[weightOff+i];
                 }
                 weightOff+=INPUTS;
             }
          }
          else
             weightOff+=FX*INPUTS;
       }
       dest[destOff+o] = ACTIVATION(sum);
    }
};
#endif




#ifdef CONV2D_3x3



inline float dotRow(const __global float *src, const __global float *weight, int n)
{
   float4 sum4 = (float4)(0.0f);
   const __global float4 *s4 = (const __global float4 *)src;
   const __global float4 *w4 = (const __global float4 *)weight;
   int fours = n>>2;
   for(int i=0;i<fours;i+=2)
   {
      sum4 = mad(s4[i], w4[i], sum4);
      sum4 = mad(s4[i+1], w4[i+1], sum4);
   }

   float2 sum2 = sum4.s01 + sum4.s23;
   return sum2.s0 + sum2.s1;

}

inline float4 dotRow4(const __global float4 *s4, const __global float4 *w4, int n)
{
   float4 sum4 = (float4)(0.0f);
   for(int i=0;i<n;i+=2)
   {
      sum4 = mad(s4[i], w4[i], sum4);
      sum4 = mad(s4[i+1], w4[i+1], sum4);
   }
   return sum4;
}


inline float dotRowLocal(const __local float *src, const __global float *w, int n)
{
   float sum = 0;
   int eights = n>>3;
   for(int i=0;i<eights;i++)
   {
      sum+= src[0]*w[0] +
            src[1]*w[1] +
            src[2]*w[2] +
            src[3]*w[3] +
            src[4]*w[4] +
            src[5]*w[5] +
            src[6]*w[6] +
            src[7]*w[7];
      src+=8;
      w+=8;
   }
   return sum;
}



inline void processTiles(const bool IsX0, const int MaxPix,
   const __global float* s0,
   const __global float* s1,
   const __global float* s2,
   const __global float *weights,
   const __global float *bias,
   __global float *out)
{
   float4 zero = 0.0f;

   // tid = threadId (0...OVEC) used to do 2*OVEC outputs at the same time
   const int tid = get_local_id(0);


   // float4 seem to be fast that float8 on hardware
   for(int o=0;o<OUTPUTS;o+= THREADS*2 )
   {
      // Each thread handles 6 horizontal outputs for 1 output
      float4 sum00 = { bias[o+tid*2], 0.0f, 0.0f, 0.0f };
      float4 sum01 = sum00;
      float4 sum02 = sum00;
      float4 sum03 = sum00;
      float4 sum04 = sum00;
      float4 sum05 = sum00;
      float4 sum10 = { bias[o+tid*2+1], 0.0f, 0.0f, 0.0f };
      float4 sum11 = sum10;
      float4 sum12 = sum10;
      float4 sum13 = sum10;
      float4 sum14 = sum10;
      float4 sum15 = sum10;

      for(int line=0;line<3;line++)
      {
         __global const float4 *s4 = (__global const float4 *)(line==0 ? s0 : line==1 ? s1 : s2 );
         if (!s4) continue;

         // Each thread is running a different output
         const __global float4 *w4 = (const __global float4 *)( weights + (o+tid*2)*INPUTS*9 + line*INPUTS*3 );

         for(int sx=0; sx<INPUTS4 ; sx++, s4++)
         {
            #define GetSrcOffset(Pixel) (Pixel<MaxPix) ?  s4[ Pixel*INPUTS4 ] : zero;

            float4 weight00 = w4[0];
            float4 weight01 = w4[INPUTS4];
            float4 weight02 = w4[INPUTS4*2];

            float4 srcA = (IsX0) ? zero : s4[-INPUTS4];
            float4 srcB = s4[0];
            float4 srcC = GetSrcOffset(1);

            float4 weight10 = w4[INPUTS4*9];
            float4 weight11 = w4[INPUTS4*10];
            float4 weight12 = w4[INPUTS4*11];
            w4++;

            sum00 = mad(weight00, srcA, sum00 );
            sum00 = mad(weight01, srcB, sum00 );
            sum00 = mad(weight02, srcC, sum00 );

            sum10 = mad(weight10, srcA, sum10 );
            sum10 = mad(weight11, srcB, sum10 );
            sum10 = mad(weight12, srcC, sum10 );

            srcA = GetSrcOffset(2);
            sum01 = mad(weight00, srcB, sum01 );
            sum01 = mad(weight01, srcC, sum01 );
            sum01 = mad(weight02, srcA, sum01 );

            sum11 = mad(weight10, srcB, sum11 );
            sum11 = mad(weight11, srcC, sum11 );
            sum11 = mad(weight12, srcA, sum11 );

            if (MaxPix==1) continue;

            srcB = GetSrcOffset(3);
            sum02 = mad(weight00, srcC, sum02 );
            sum02 = mad(weight01, srcA, sum02 );
            sum02 = mad(weight02, srcB, sum02 );

            sum12 = mad(weight10, srcC, sum12 );
            sum12 = mad(weight11, srcA, sum12 );
            sum12 = mad(weight12, srcB, sum12 );

            if (MaxPix==2) continue;
            srcC = GetSrcOffset(4);
            sum03 = mad(weight00, srcA, sum03 );
            sum03 = mad(weight01, srcB, sum03 );
            sum03 = mad(weight02, srcC, sum03 );

            sum13 = mad(weight10, srcA, sum13 );
            sum13 = mad(weight11, srcB, sum13 );
            sum13 = mad(weight12, srcC, sum13 );

            if (MaxPix==3) continue;
            srcA = GetSrcOffset(5);
            sum04 = mad(weight00, srcB, sum04 );
            sum04 = mad(weight01, srcC, sum04 );
            sum04 = mad(weight02, srcA, sum04 );

            sum14 = mad(weight10, srcB, sum14 );
            sum14 = mad(weight11, srcC, sum14 );
            sum14 = mad(weight12, srcA, sum14 );

            if (MaxPix==4) continue;
            srcB = GetSrcOffset(6);
            sum05 = mad(weight00, srcC, sum05 );
            sum05 = mad(weight01, srcA, sum05 );
            sum05 = mad(weight02, srcB, sum05 );

            sum15 = mad(weight10, srcC, sum15 );
            sum15 = mad(weight11, srcA, sum15 );
            sum15 = mad(weight12, srcB, sum15 );
         }
      }


      // Write  output for 6 pixels per thread + bias + activation

      float2 f2;
      float val;
      #define SAVE(X,Y) \
         if (X<MaxPix) { \
         f2 = sum##Y##X.xy + sum##Y##X.zw; \
         val = f2.x + f2.y; \
         out[(tid*2)+Y+OUTPUTS*X] = ACTIVATION(val); \
         }

      SAVE(0,0)
      SAVE(0,1)
      SAVE(1,0)
      SAVE(1,1)
      SAVE(2,0)
      SAVE(2,1)
      SAVE(3,0)
      SAVE(3,1)
      SAVE(4,0)
      SAVE(4,1)
      SAVE(5,0)
      SAVE(5,1)

      out += THREADS*2;
   }
}



__kernel void Conv2D(const __global float* src, __global float *dest, const __global float *weights, const __global float *bias, const int width, const int height)
{
   const int tid = get_local_id(0);
   const int row = get_global_id(1);

   const __global float *s0 = row>0 ? src + (row-1)*INPUTS*width:0;
   const __global float *s1 =  src + (row)*INPUTS*width;
   const __global float *s2 = row<height-1 ? src + (row+1)*INPUTS*width: 0;

   __global float *out = dest + row*OUTPUTS*width;

   if (width<7)
   {
      processTiles(true, width, s0, s1, s2, weights, bias, out);
   }
   else
   {
      processTiles(true, 1000, s0, s1, s2, weights, bias, out);
      out += OUTPUTS*6;
      if (s0) s0+=INPUTS*6;
      s1+=INPUTS*6;
      if (s2) s2+=INPUTS*6;

      int x = 6;
      for(  ;x+6<width; x+=6 )
      {
         processTiles(false, 1000, s0, s1, s2, weights, bias, out);

         out += OUTPUTS*6;
         if (s0) s0+=INPUTS*6;
         s1+=INPUTS*6;
         if (s2) s2+=INPUTS*6;
      }

      processTiles(false, width-x, s0, s1, s2, weights, bias, out);
   }
}

#endif



