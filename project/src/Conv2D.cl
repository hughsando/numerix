
#define SRC 32
#define PIX 32
#define OUT 32


#ifdef CONV2D_1x1
__kernel void Conv2D(const __global float* inSrc, const __global float *inWeights, const __global float *inBias, const int width, const int height, __global float *dest) {
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
};
#endif


#ifdef CONV2D_SIMPLE

#define dMin (-FX/2)
#define dMax (FX+dMin)

__kernel void Conv2D(const __global float* src, const __global float *weights, const __global float *inBias, const int width, const int height, __global float *dest ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int srcStride = width*INPUTS;
    
    if (x<width && y<height) {
    int weightOff = 0;
    int destOff = (y*width+x) * OUTPUTS;
    for(int o=0;o<OUTPUTS;o++) {
       float sum=inBias[o];
       for(int dy=dMin; dy<dMax; dy++) {
          int sy = y+dy;
          if (sy>=0 && sy<height) {
             for(int dx=dMin; dx<dMax; dx++) {
                 int sx = x+dx;
                 if (sx>=0 && sx<width) {
                    int srcOff = sy*srcStride + INPUTS*sx;
                    for(int i=0;i<INPUTS;i++)
                        sum += src[srcOff+i] * weights[weightOff+i];
                 }
                 weightOff+=INPUTS;
             }
          } else {
             weightOff += INPUTS * (dMax-dMin);
          }
       }
    dest[destOff+o] = ACTIVATION(sum);
    }
   }
};
#endif




#ifdef CONV2D_3x3


#define OVEC 32

#ifdef INTEL
inline float dotRow(const __global float *src, const __global float *weight, int n)
{
   float8 sum8 = (float8)(0.0f);
   const __global float8 *s8 = (const __global float8 *)src;
   const __global float8 *w8 = (const __global float8 *)weight;
   int eights = n>>3;
   for(int i=0;i<eights;i++)
      sum8 = mad(s8[i], w8[i], sum8);

   float4 sum4 = sum8.s0123 + sum8.s4567;
   float2 sum2 = sum4.s01 + sum4.s23;
   return sum2.s0 + sum2.s1;
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

#endif

__kernel void Conv2D(const __global float* src, const __global float *weights, const __global float *bias, const int width, const int height, __global float *dest)
{
   #ifdef INTEL
   const int tid = get_local_id(0);
   const int row = get_global_id(1);

   const __global float *s0 = row>0 ? src + (row-1)*INPUTS*width : 0;
   const __global float *s1 =  src + (row)*INPUTS*width;
   const __global float *s2 = row<height-1 ? src + (row+1)*INPUTS*width: 0;

   __global float *out = dest + row*OUTPUTS*width;



   // First pixel...
   const __global float *w0 = weights + INPUTS + tid*INPUTS*9;

   for(int o=0;o<OUTPUTS;o+=OVEC)
   {
      float sum = bias[o+tid];
      if (s0)
         sum += dotRow( s0, w0, INPUTS*2 );
      sum += dotRow( s1, w0 + INPUTS*3, INPUTS*2 );
      if (s2)
         sum += dotRow( s2, w0 + INPUTS*6, INPUTS*2 );

      w0 += INPUTS*9*OVEC;

      out[o+tid] = ACTIVATION(sum);
   }
   out += OUTPUTS;

   // Middle pixels...
   if (s0 && s2)
   {
      for(int x=1;x<width-1;x++)
      {
         w0 = weights + tid*INPUTS*9;

         #if 0

         __local float srcBuf[INPUTS*9];
         for(int i=0;i<INPUTS*3;i+=OVEC)
         {
            srcBuf[i+tid] = s0[i+tid];
            srcBuf[i+tid+INPUTS*3] = s1[i+tid];
            srcBuf[i+tid+INPUTS*6] = s2[i+tid];
         }
         barrier(CLK_LOCAL_MEM_FENCE);

         w0 = weights;
         for(int o=0;o<OUTPUTS;o+=OVEC)
         {
            float sum = bias[o+tid] + dotRowLocal( srcBuf, w0 + tid*INPUTS*9,  INPUTS*9 );
            w0 += INPUTS*9*OVEC;
            out[o+tid] = ACTIVATION(sum);
         }
         barrier(CLK_LOCAL_MEM_FENCE);


         #else

         for(int o=0;o<OUTPUTS;o+=OVEC)
         {
            float sum = bias[o+tid] +
               dotRow( s0, w0,  INPUTS*3 ) +
               dotRow( s1, w0 + INPUTS*3, INPUTS*3 ) +
               dotRow( s2, w0 + INPUTS*6, INPUTS*3 );

            w0 += INPUTS*9*OVEC;
            out[o+tid] = ACTIVATION(sum);
         }
         #endif

         out += OUTPUTS;
         s0+=INPUTS;
         s1+=INPUTS;
         s2+=INPUTS;
      }
   }
   else
   {
      for(int x=1;x<width-1;x++)
      {
         w0 = weights + tid*INPUTS*9;
         for(int o=0;o<OUTPUTS;o+=OVEC)
         {
            float sum = bias[o+tid];
            if (s0)
               sum += dotRow( s0, w0, INPUTS*3 );
            sum += dotRow( s1, w0+INPUTS*3 , INPUTS*3 );
            if (s2)
               sum += dotRow( s2, w0 + INPUTS*6, INPUTS*3 );

            w0 += INPUTS*9*OVEC;

            out[o+tid] = ACTIVATION(sum);
         }

         out += OUTPUTS;
         if (s0) s0+=INPUTS;
                 s1+=INPUTS;
         if (s2) s2+=INPUTS;
      }
   }

   // last pixel
   w0 = weights + tid*INPUTS*9;
   for(int o=0;o<OUTPUTS;o+=OVEC)
   {
      float sum = bias[o+tid];
      if (s0)
         sum += dotRow( s0, w0, INPUTS*2 );
      sum += dotRow( s1, w0 + INPUTS*3, INPUTS*2 );
      if (s2)
         sum += dotRow( s2, w0 + INPUTS*6, INPUTS*2 );
      w0 += INPUTS*9*OVEC;

      out[o+tid] = ACTIVATION(sum);
   }



   #else
   int pixels = width*height;
   int pix0 = PIX * ( ((pixels+PIX-1)/PIX)*get_group_id(0)/get_num_groups(0) );
   int maxPix = PIX * ( ((pixels+PIX-1)/PIX)*(get_group_id(0)+1)/get_num_groups(0) );
   if (maxPix>pixels)
      maxPix = pixels;

   const int pixId = get_local_id(0) / PIX;
   const int chanId = get_local_id(0) % PIX;

   for( ; pix0<maxPix; pix0+=PIX)
   {
      // Calculate src offset and range for pixel in this thread
      int pix = pix0 + pixId;
      int py = pix / width; // todo - fast version?
      int px = pix - py*width;

      for(int oBase=0;oBase<OUTPUTS;oBase+=OUT)
      {
         __local float outputSum[PIX][OUT];
         // Initialize outputSum
         outputSum[pixId][chanId] = bias[oBase+chanId];
         barrier(CLK_LOCAL_MEM_FENCE);

         int weightBase = oBase * (INPUTS * 9);
         for(int sy=0; sy<3; sy++)
         {
            int srcBase = ((py+(sy-1))*width + px - 1) *INPUTS;
            for(int sx=0; sx<3; sx++)
            {
               int valid = (sx>0|px>0) &
                           (sy>0|py>0) &
                           (sx<2|px<(width-1)) &
                           (sy<2|py<(height-1)) &
                           (pix<maxPix);
               for(int ch0 = 0; ch0 < INPUTS; ch0+=SRC)
               {
                  __local float srcBuf[PIX][SRC];
                  __local float wBuf[OUT][SRC];

                  // Fill Src ...
                  srcBuf[pixId][chanId] = valid ? src[ srcBase + ch0 + chanId ] : 0.0f;

                  // Fill weights ...
                  //  w = out0:  s0 s1 s2 s3 .... s31
                  //      out2:  s0 s1 s2 s3 .... s31
                  //        ...
                  //      outN:  s0 s1 s2 s3 .... s31
                  wBuf[pixId][chanId] = weights[ weightBase + pixId*(INPUTS*9) + chanId ];

                  barrier(CLK_LOCAL_MEM_FENCE);

                  weightBase += SRC;

                  // chanId == outId
                  float sum = outputSum[pixId][chanId];
                  for(int s =0; s < SRC; s ++)
                     sum += srcBuf[pixId][s] * wBuf[chanId][s];

                  outputSum[pixId][chanId] = sum;

                  barrier(CLK_LOCAL_MEM_FENCE);
               }
               srcBase += INPUTS;
            }
         }

         if (pix < maxPix)
            dest[(py*width + px)*OUTPUTS + oBase+chanId] = ACTIVATION( outputSum[pixId][chanId] );

         barrier(CLK_LOCAL_MEM_FENCE);
      }
   }
   #endif
}
#endif



