
#define SRC 32
#define PIX 32
#define OUT 32


__kernel void Conv2D(const __global float* src, const __global float *weights, const __global float *bias, const int srcW, const int srcH, const int inputs, __global float *dest, const int outputs)
{
   int pixels = srcW*srcH;
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
      int py = pix / srcW; // todo - fast version?
      int px = pix - py*srcW;

      for(int oBase=0;oBase<outputs;oBase+=OUT)
      {
         __local float outputSum[PIX][OUT];
         // Initialize outputSum
         outputSum[pixId][chanId] = bias[oBase+chanId];
         barrier(CLK_LOCAL_MEM_FENCE);

         int weightBase = oBase * inputs * 9;
         for(int sy=0; sy<3; sy++)
         {
            int srcBase = ((py+(sy-1))*srcW + px - 1) *inputs;
            for(int sx=0; sx<3; sx++)
            {
               int valid = (sx>0|px>0) &
                           (sy>0|py>0) &
                           (sx<2|px<(srcW-1)) &
                           (sy<2|py<(srcH-1)) &
                           (pix<maxPix);
               for(int ch0 = 0; ch0 < inputs; ch0+=SRC)
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
                  wBuf[pixId][chanId] = weights[ weightBase + pixId*inputs*9 + chanId ];

                  barrier(CLK_LOCAL_MEM_FENCE);

                  weightBase += SRC;

                  // chanId == outId
                  float sum = outputSum[pixId][chanId];
                  for(int s =0; s < SRC; s ++)
                     sum += srcBuf[pixId][s] * wBuf[chanId][s];

                  outputSum[pixId][chanId] = sum;

                  barrier(CLK_LOCAL_MEM_FENCE);
               }
               srcBase += inputs;
            }
         }

         if (pix < maxPix)
            dest[(py*srcW + px)*outputs + oBase+chanId] = ACTIVATION( outputSum[pixId][chanId] );

         barrier(CLK_LOCAL_MEM_FENCE);
      }
   }
}



