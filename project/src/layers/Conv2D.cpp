#include <Tensor.h>
#include <Layer.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

namespace numerix
{

// ----- Conv2DBase ---------------

Conv2DBase::Conv2DBase(int inStrideY, int inStrideX,
       Activation inActivation, Padding inPadding,
       Tensor *inWeights, Tensor *inPWeights, Tensor *inBias)
{
   strideY = inStrideY;
   strideX = inStrideX;
   activation = inActivation;
   padding = inPadding;
   diSize = 0;
   padInputsWithZero = false;
   weightsOriginal = 0;

   // Output x Height x Width x Input
   CShape s = inWeights->shape;
   if (!inPWeights && s.size()!=4)
      TensorThrow("Invalid Conv2D weight shape");
   if (inPWeights && s.size()!=3)
      TensorThrow("Invalid SeparableConv2D depthwise shape");

   if (inPWeights)
   {
      CShape p = inPWeights->shape;
      if (p.size()!=2)
         TensorThrow("Invalid SeparableConv2D pointwise shape");
      if (s[0]!=p[1])
         TensorThrow("SeparableConv2D mismatched weights");
      diSize = p[1];

      outputs = p[0];
      inputs =  s[0];
      filterY = s[1];
      filterX = s[2];
      if (filterX==1 && filterY==1)
         TensorThrow("SeparableConv2D 1x1 not supported");
   }
   else
   {
      outputs = s[0];
      filterY = s[1];
      filterX = s[2];
      inputs =  s[3];
   }

   if (inBias && inBias->shape.size()!=1)
      TensorThrow("Conv2D - bias should be one dimensional");

   if (inBias && inBias->shape[0]!=outputs)
   {
      char buf[1000];
      sprintf(buf, "Conv2D - bias size (%d) does not match output size(%d)", inBias->shape[0], outputs);
      TensorThrow(buf);
   }


   weights = inWeights->incRef();
   pweights = inPWeights ? inPWeights->incRef() : 0;
   bias = inBias ? inBias->incRef() : 0;

   is1x1 = filterX*filterY==1;
   is1x1Aligned = is1x1 && (inputs & 0x3) == 0;

}


Conv2DBase::~Conv2DBase()
{
   if (weightsOriginal)
      weightsOriginal->decRef();
   weights->decRef();
   if (pweights)
      pweights->decRef();
   if (bias)
      bias->decRef();
}


void Conv2DBase::setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars)
{
   // a set of output features are first "normalized":
   //
   // O'i = (Oi - Mean_i)/(sqrt(Vars_i) +  .000001f)
   //
   // Then 'scale_bias'
   //
   // O''i = O'i * Scale_i
   //
   // Let Ki = Scale_i/(sqrt(Vars_i) +  .000001f)
   //     Ci = -Mean_i * Ki
   //
   // O' = Ki Oi + Ci
   // This is the same as multiplying the Weights_i by Ki and adding Ci to the biases

   const float *scale = (const float *)inScales->cpuRead();
   const float *mean = (const float *)inMeans->cpuRead();
   const float *var = (const float *)inVars->cpuRead();

   if (!bias)
   {
      Shape s(1);
      s[0]=outputs;
      bias = new Tensor(Float32, s);
      bias->zero(0,outputs);
   }
   float *b = (float *)bias->cpuWritePart();
   float *w = (float *)weights->cpuWritePart();

   int wCount = weights->strides[0];
   for(int i=0;i<outputs;i++)
   {
      float Ki = scale[i]/(sqrt(var[i])+.000001f);
      for(int j=0;j<wCount;j++)
         *w++ *= Ki;
      b[i] -= mean[i]*Ki;
   }

   rebuildWeights();
}


void Conv2DBase::reduceInputs(int inCount)
{
   if (!weightsOriginal)
      weightsOriginal = weights;
   else
      weights->decRef();

   // take last ones ...
   int skip = weightsOriginal->shape[3] - inCount;
   weights = weightsOriginal->resizeAxis(3,inCount,skip);
   inputs = inCount;
   rebuildWeights();
}


Tensor *Conv2DBase::run(Tensor *inSrc0, Tensor *inBuffer)
{
   if (inSrc0->type != Float32)
      TensorThrow("Conv2D only supports Float32 tensors");

   CShape sin = inSrc0->shape;
   if (sin.size()!=3)
   {
      printf("Conv2D %dx%dx%dx%d\n", outputs, filterY, filterX, inputs);
      printf("Src dimension %d :", (int)sin.size());
      for(int i=0;i<sin.size();i++)
         printf(" %d", sin[i]);
      printf("\n");
      TensorThrow("Conv2D only supports H*W*C tensors");
   }

   if (sin[2]!=inputs && padInputsWithZero)
   {
      int maxIn = weightsOriginal ? weightsOriginal->shape[3] : inputs;
      if (sin[2]>maxIn)
         TensorThrow("Conv2D - too many inputs for the number of weights");
      reduceInputs(sin[2]);
   }

   if (sin[2]!=inputs)
   {
      printf("sin : %d %d %d\n", sin[0], sin[1], sin[2]);
      printf("weights : %d %dx%d %d\n", outputs, filterY, filterX, inputs );
      TensorThrow("Conv2D - weights do not match the number of input channels");
   }

   srcH = sin[0];
   srcW = sin[1];

   destH = 0;
   destW = 0;
   padOx = 0;
   padOy = 0;

   if (padding==padSame)
   {
      destW = (srcW+strideX-1)/strideX;
      //int padX = (destW - 1)*strideX + filterX - srcW;
      // This is pad-left, right does not really matter
      padOx = (filterX-1)>>1;

      destH = (srcH+strideY-1)/strideY;
      //int padY = (destH - 1)*strideY + filterY - srcH;
      padOy = (filterY-1)>>1;
   }
   else // padValid
   {
      destW = (srcW-filterX+1 + strideX-1)/strideX;
      destH = (srcH-filterY+1 + strideY-1)/strideY;
   }

   bool match = false;
   if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
   {
      CShape s = inBuffer->shape;
      match = s[0]==destH && s[1]==destW && s[2]==outputs;
   }
   Tensor *result = inBuffer;
   if (!match)
   {
      Shape s(3);
      s[0] = destH;
      s[1] = destW;
      s[2] = outputs;
      result = new Tensor( Float32, s );
   }
   //printf("Cov2d -> %d %d %d\n", destW, destH, outputs);

   doRun(inSrc0, result);

   return result;
}


// -- Conv2D -----------------------


class Conv2D : public Conv2DBase
{
   bool       interlacedWeights;
   int        interlacedCount;
   std::vector <float *> srcBuffers;
   std::vector <float *> diBuffers;
   std::vector <float *> weightBuffers;

   float      *alignedWeightsBuffer;
   float      *alignedWeights;
   float      *alignedBias;
   int        alignedWeightSize;


public:
   Conv2D(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inPWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inActivation, inPadding,  inWeights, inPWeights, inBias)
   {
      interlacedWeights = false;
      interlacedCount = 0;
      alignedBias = 0;
      alignedWeights = 0;

      #ifdef NUMERIX_SIMD
      interlacedWeights = (outputs & 0x3)==0  && !pweights && (!is1x1 || is1x1Aligned);
      #endif

      rebuildWeights();
   }


   void setPadInput()
   {
      padInputsWithZero = true;
   }


   void rebuildWeights()
   {
      releaseFloats();
      srcBuffers.resize(0);
      diBuffers.resize(0);
      weightBuffers.resize(0);


      alignedBias = bias ? (float *)bias->cpuRead() : 0;

      alignedWeightsBuffer = 0;
      alignedWeights = (float *)weights->cpuRead();
      alignedWeightSize = filterX*filterY*inputs;

      if ( !pweights && (alignedWeightSize & 0x3) && outputs>1 )
      {
         alignedWeightSize = (alignedWeightSize + 3) & ~3;
         alignedWeightsBuffer = allocFloats(alignedWeightSize*outputs);
         int wSize = filterX*filterY*inputs;
         for(int o=0;o<outputs;o++)
         {
            memcpy(alignedWeightsBuffer + o*alignedWeightSize, alignedWeights + o*wSize, wSize*sizeof(float));
         }
         alignedWeights = alignedWeightsBuffer;
      }

      if (is1x1)
      {
         if (!is1x1Aligned)
         {
            int wBytes = weights->elementCount * sizeof(float);

            for(int i=1;i<4;i++)
            {
               float *buffer = allocFloats( weights->elementCount - i);
               memcpy(buffer, weights->cpuRead() + sizeof(float)*i, wBytes - sizeof(float)*i);
               srcBuffers.push_back(buffer);
            }
         }
      }
      else
      {
         int threads = GetWorkerCount();
         int paddedSize = (filterX*filterY*inputs + 3) & ~0x3;
         for(int i=0;i<threads;i++)
         {
            srcBuffers.push_back( allocFloats(paddedSize,true) );
            if (diSize)
               diBuffers.push_back( (float *)allocFloats(diSize) );
         }
      }

      if (interlacedWeights)
         createInterlacedWeights();
   }


   void createInterlacedWeights()
   {
      if (!alignedBias)
      {
         alignedBias = allocFloats(outputs);
         memset(alignedBias,0,outputs*sizeof(float));
      }
      int count = weights->strides[0];
      int paddedSize = (count + 3) & ~0x3;
      int odd = count & 0x3;
      interlacedCount = paddedSize>>2;
      const float *wSrc = (float *)weights->cpuRead();
      for(int o=0; o<outputs; o+=4)
      {
         float *interlacedBuf = allocFloats( paddedSize * 4, true );
         float *b = interlacedBuf;
         const float *w0 = wSrc; wSrc+=count;
         const float *w1 = wSrc; wSrc+=count;
         const float *w2 = wSrc; wSrc+=count;
         const float *w3 = wSrc; wSrc+=count;
         int fours = count>>2;
         for(int i=0;i<fours;i++)
         {
            *b++ = *w0++;
            *b++ = *w0++;
            *b++ = *w0++;
            *b++ = *w0++;

            *b++ = *w1++;
            *b++ = *w1++;
            *b++ = *w1++;
            *b++ = *w1++;

            *b++ = *w2++;
            *b++ = *w2++;
            *b++ = *w2++;
            *b++ = *w2++;

            *b++ = *w3++;
            *b++ = *w3++;
            *b++ = *w3++;
            *b++ = *w3++;
         }
         if (odd)
         {
            *b++ = *w0++;
            *b++ = odd<2 ? 0 : *w0++;
            *b++ = odd<3 ? 0 : *w0;
            *b++ = 0;

            *b++ = *w1++;
            *b++ = odd<2 ? 0 : *w1++;
            *b++ = odd<3 ? 0 : *w1;
            *b++ = 0;

            *b++ = *w2++;
            *b++ = odd<2 ? 0 : *w2++;
            *b++ = odd<3 ? 0 : *w2;
            *b++ = 0;

            *b++ = *w3++;
            *b++ = odd<2 ? 0 : *w3++;
            *b++ = odd<3 ? 0 : *w3;
            *b++ = 0;
         }

         weightBuffers.push_back(interlacedBuf);
      }
   }

   Tensor *src0;
   Tensor *destTensor;

   virtual void doRun(Tensor *input, Tensor *output)
   {
      startRun();
      src0 = input;
      destTensor = output;
      src0->cpuRead();
      destTensor->cpuWrite();

      runThreaded();
      src0 = 0;
      destTensor = 0;
      endRun();
   }



   void runThread(int threadId)
   {
      if (is1x1)
         runThread1x1(threadId);
      else
         runThreadMulti(threadId);
   }

   void runThread1x1(int threadId)
   {
      const float *b = alignedBias;

      const float *w0 = alignedWeights;
      const float *w1 = is1x1Aligned ? w0 : srcBuffers[0];
      const float *w2 = is1x1Aligned ? w0 : srcBuffers[1];
      const float *w3 = is1x1Aligned ? w0 : srcBuffers[2];

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;

         const float *src = (const float *)src0->cpuRead() + srcW*inputs*y*strideY;
         float *dest = (float *)destTensor->cpuWrite() + destW*outputs*y;
         for(int x=0;x<destW;x++)
         {
            if (interlacedWeights)
            {
               int bid = 0;
               for(int o=0;o<outputs;o+=4)
               {
                  dot4Interlaced(dest, alignedBias+bid*4, weightBuffers[bid], src, interlacedCount, activation);
                  bid++;
                  dest+=4;
               }
            }
            else
            {
               const float *w;
               switch( (size_t)src & 0x3 )
               {
                  case 0:
                     w = w0;
                     for(int o=0;o<outputs;o++)
                     {
                        float sum = dot(b?b[o] : 0.0f, w, src, inputs, activation);
                        *dest++ = sum;
                        w+=alignedWeightSize;
                     }
                     break;
                  case 1:
                     w = w3;
                     for(int o=0;o<outputs;o++)
                     {
                        float sum = b?b[o] : 0.0f;
                        sum += w0[0]*src[0] + w0[1]*src[1] + w0[2]*src[2];
                        sum = dot(sum, w, src+3, inputs-3, activation);
                        *dest++ = sum;
                        w+=alignedWeightSize;
                     }
                     break;
                  case 2:
                     w = w2;
                     for(int o=0;o<outputs;o++)
                     {
                        float sum = b?b[o] : 0.0f;
                        sum += w0[0]*src[0] + w0[1]*src[1];
                        sum = dot(sum, w, src+2, inputs-2, activation);
                        *dest++ = sum;
                        w+=alignedWeightSize;
                     }
                     break;
                  case 3:
                     w = w1;
                     for(int o=0;o<outputs;o++)
                     {
                        float sum = b?b[o] : 0.0f;
                        sum += w0[0]*src[0];
                        sum = dot(sum, w, src+1, inputs-1, activation);
                        *dest++ = sum;
                        w+=alignedWeightSize;
                     }
                     break;

               }
            }
            src+=inputs*strideX;
         }
      }
   }

   void runThreadMulti(int threadId)
   {
      const float *b = bias ? (const float *)bias->cpuRead() : 0;
      const int *srcStride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];
      float *di = pweights ? &diBuffers[threadId][0] : 0;

      int featureSize = filterX*filterY*inputs;
      int filters = filterX*filterY;

      int filterW = filterX*inputs;
      int filterFours = filterW>>2;
      int filterSixteens = filterW>>4;
      float *srcPtr = &srcBuffers[threadId][0];
      int filterRow = filterW*sizeof(float);

      const float *sIn = (float *)src0->cpuRead();

      while(true)
      {
         int y = getNextJob();
         if (y>=destH)
            break;


         float *dest = (float *)destTensor->cpuWrite() + destW*outputs*y;
         int srcY = y*strideY;

         int dyMin = std::max(padOy-srcY,0);
         int dyMax = std::min(srcH+padOy-srcY,filterY);

         if (dyMin>0)
            memset(srcPtr,0,filterRow*dyMin);

         if (dyMax<filterY)
            memset(srcPtr+dyMax*filterW,0,filterRow*(filterY-dyMax) );

         for(int x=0;x<destW;x++)
         {
            int srcX = x*strideX;

            int dxMin = std::max(padOx-srcX,0);
            int dxMax = std::min(srcW+padOx-srcX,filterX);
            int xRange = dxMax-dxMin;

            const float *s = sIn + (srcY+dyMin-padOy)*srcStride[0] + (srcX+dxMin-padOx)*srcStride[1];

            for(int dy=dyMin;dy<dyMax;dy++)
            {
               float *sp = srcPtr + filterW*dy;

               if (dxMin>0)
               {
                  memset(sp, 0, dxMin*inputs*sizeof(float));
                  sp +=dxMin*inputs;
               }

               memcpy(sp, s, xRange*inputs*sizeof(float));
               s+= srcStride[0];

               if (dxMax<filterX)
                  memset(sp + (dxMax-dxMin)*inputs, 0, (filterX-dxMax)*inputs*sizeof(float));
            }


            const float *w = alignedWeights;
            if (pweights)
            {
               for(int d=0;d<diSize;d++)
               {
                  int srcOff = d; // todo - depth_multiplier > 1
                  di[d] = dotSkip(w, srcPtr+srcOff, filters, inputs);
                  w+=filters;
               }

               const float *w = (const float *)pweights->cpuRead();
               for(int o=0;o<outputs;o++)
               {
                  float sum = dot(b?b[o]:0.0f, w, di, diSize, activation);
                  *dest++ = sum;
                  w+=diSize;
               }
            }
            else
            {
               if (interlacedWeights)
               {
                  int bid = 0;
                  for(int o=0;o<outputs;o+=4)
                  {
                     dot4Interlaced(dest, alignedBias+bid*4, weightBuffers[bid], srcPtr, interlacedCount, activation);
                     bid++;
                     dest+=4;
                  }
               }
               else
               {
                  for(int o=0;o<outputs;o++)
                  {
                     float sum = dot(b?b[o]:0.0f, w, srcPtr, featureSize, activation);
                     *dest++ = sum;
                     w+=alignedWeightSize;
                  }
               }
            }
         }
      }
   }
};


#define NUMERIX_WINOGRAD
#ifdef NUMERIX_WINOGRAD

/*
 
 Winograd transform taken from 'nnpack':

Copyright (c) 2017 Facebook Inc.
Copyright (c) 2015-2017, Georgia Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


class Conv2DWinograd : public Conv2DBase
{
   std::vector <float *> srcBuffers;
   std::vector <float *> scratchBuffer;
   float *transformWeights;
   float *biasPtr;


public:
   Conv2DWinograd(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inActivation, inPadding,  inWeights, 0, inBias)
   {
      //printf("winograd!\n");
      rebuildWeights();
   }

   static void TransKernel(const float *src0, const float *src1, const float *src2,
      float *t0,
      float *t1,
      float *t2,
      float *t3,
      float *t4,
      float *t5,
      float *t6,
      float *t7,
      bool rescale_coefficients)
   {
      const psimd_f32 g0(src0);
      const psimd_f32 g1(src1);
      const psimd_f32 g2(src2);
      /*
       * w0 = g0
       * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
       * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
       * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
       * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
       * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
       * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
       * w7 = g2
       */

      /*
       * Compute
       *   w2 := g0 + g2
       *   w4 := g0 + 4 * g2
       *   w6 := g2 + 4 * g0
       */
      const psimd_f32 const_4 = psimd_splat_f32(4.0f);
      psimd_f32 w2 = g0 + g2;
      psimd_f32 w4 = g0 + const_4 * g2;
      psimd_f32 w6 = g2 + const_4 * g0;

      /*
       * Compute
       *   w1 = (g0 + g2) + g1
       *   w2 = (g0 + g2) - g1
       *   w3 = (g0 + 4 * g2) + 2 * g1
       *   w4 = (g0 + 4 * g2) - 2 * g1
       *   w5 = (g2 + 4 * g0) + 2 * g1
       *   w6 = (g2 + 4 * g0) - 2 * g1
       */
      const psimd_f32 two_g1 = g1 * psimd_splat_f32(2.0f);
      psimd_f32 w1 = w2 + g1;
      w2 = w2 - g1;
      psimd_f32 w3 = w4 + two_g1;
      w4 = w4 - two_g1;
      psimd_f32 w5 = w6 + two_g1;
      w6 = w6 - two_g1;

      if (rescale_coefficients) {
         //const psimd_f32 minus_2_over_9 = psimd_splat_f32(-0x1.C71C72p-3f);
         const psimd_f32 minus_2_over_9 = psimd_splat_f32(-0.2222222222222222f);
         w1 *= minus_2_over_9;
         w2 *= minus_2_over_9;

         //const psimd_f32 rcp_90 = psimd_splat_f32( 0x1.6C16C2p-7f);
         const psimd_f32 rcp_90 = psimd_splat_f32( 0.01111111111111f );
         w3 *= rcp_90;
         w4 *= rcp_90;

         //const psimd_f32 rcp_180 = psimd_splat_f32( 0x1.6C16C2p-8f);
         const psimd_f32 rcp_180 = psimd_splat_f32( 0.00555555555556f);
         w5 *= rcp_180;
         w6 *= rcp_180;
      }

      psimd_f32_store(t0, g0);
      psimd_f32_store(t1, w1);
      psimd_f32_store(t2, w2);
      psimd_f32_store(t3, w3);
      psimd_f32_store(t4, w4);
      psimd_f32_store(t5, w5);
      psimd_f32_store(t6, w6);
      psimd_f32_store(t7, g2);
   }



   void rebuildWeights()
   {
      releaseFloats();

      srcBuffers.resize(0);
      scratchBuffer.resize(0);

      int threads = GetWorkerCount();

      for(int i=0;i<threads;i++)
      {
         srcBuffers.push_back( allocFloats(8*8*inputs,false) );
         scratchBuffer.push_back( allocFloats(8*8*4,false) );
      }


      // Transform O*3*3*I weights into
      //
      // (O/4)*8*8*I4o winograd coeffs...

      // Interlace 4 outputs ...
      /*
       W 0:3   0,0                     0,1                 ...  7,7
       w0: i0 i1 i2 i3             w0: i4 i5 i6 i7
          w1: i0 i1 i2 i3           w1: i4 i5 i6 i7
             w2: i0 i1 i2 i3          w2: i4 i5 i6 i7
                w3: i0 i1 i2 i3         w3: i4 i5 i6 i7

       W 4:7   0,0                     0,1                 ...  7,7

       ...
      */


      const float *weightIn = (const float *)weights->cpuRead();
      transformWeights = allocFloats( 8*8*inputs*outputs, true );

      // Transform inputs, to (O/4)WHi4o
      // output stride between pixel
      int os = inputs*4;
      for(int o=0;o<outputs;o++)
      {
         int baseBlock = (o>>2)*8*8*4*inputs;
         int baseOffset = o&3;
         for(int inputIdx=0;inputIdx<inputs;inputIdx+=4)
         {
            const float *w0 = weightIn + o*3*3*inputs + inputIdx;
            float *t = scratchBuffer[0];

            // Trans kernel columns of 4 inputs into scratch buffer
            for(int col=0;col<3;col++)
            {
               TransKernel(w0, w0+inputs*3, w0+inputs*6, \
                        t, t+3*4, t+3*4*2, t+3*4*3, t+3*4*4, t+3*4*5, t+3*4*6, t+3*4*7, true);
               w0 += inputs;
               t+=4;
            }
            // Trans scratch buffer rows to 8*8 lots of 4 inputs and write to interlaced-weights buffer
            float *w = scratchBuffer[0];

            float *out = transformWeights + baseBlock + inputIdx*4 + baseOffset*4;
            for(int row=0;row<8;row++)
            {
               TransKernel(w, w+4, w+8,
                           out, out+os, out+os*2, out+os*3, out+os*4, out+os*5, out+os*6, out+os*7, true );
               w+=3*4;
               out += os*8;
            }
         }
      }


      if (bias)
         biasPtr = (float *)bias->cpuRead();
      else
         biasPtr = allocFloats(outputs,true);
   }


   Tensor *src0;
   Tensor *destTensor;

   virtual void doRun(Tensor *input, Tensor *output)
   {
      startRun();
      src0 = input;
      destTensor = output;
      src0->cpuRead();
      destTensor->cpuWrite();

      runThreaded();

      src0 = 0;
      destTensor = 0;
      endRun();
   }

   inline void runTile( float *src, float *dest,
                        int xCount, int yCount, 
                        float *scratch)
   {

      #define TRANS_WINO(i0, i1, i2, i3, i4, i5, i6, i7, \
                         o0, o1, o2, o3, o4, o5, o6, o7 ) { \
         psimd_f32 d0(i0); \
         psimd_f32 d1(i1); \
         psimd_f32 d2(i2); \
         psimd_f32 d3(i3); \
         psimd_f32 d4(i4); \
         psimd_f32 d5(i5); \
         psimd_f32 d6(i6); \
         psimd_f32 d7(i7); \
         /*  Compute wd0 := d0 - d6  */ \
         psimd_f32 wd0 = d0 - d6; \
         const psimd_f32 d4_sub_d2 = d4 - d2; \
         /*  Compute wd7 := d7 - d1  */ \
         psimd_f32 wd7 = d7 - d1; \
         const psimd_f32 d3_sub_d5 = d3 - d5; \
         /*  Compute wd1 := d2 + d6  */ \
         psimd_f32 wd1 = d2 + d6; \
         /*  Compute wd2 := d1 + d5  */ \
         psimd_f32 wd2 = d1 + d5; \
         /*  Compute wd4 := d5 + 0.25 * d1  */ \
         const psimd_f32 const_0_25 = psimd_splat_f32(0.25f); \
         psimd_f32 wd4 = d5 + const_0_25 * d1; \
         /*  Compute wd5 := d6 - 5.0 * d4  */ \
         psimd_f32 wd5 = d6 - psimd_splat_f32(5.0f) * d4; \
         /*  Compute wd3 := d6 + 0.25 * d2  */ \
         psimd_f32 wd3 = d6 + const_0_25 * d2; \
         /*  Compute wd6 := d1 + 0.25 * d5  */ \
         psimd_f32 wd6 = d1 + const_0_25 * d5; \
 \
         const psimd_f32 const_5_25 = psimd_splat_f32(5.25f); \
         /*  Compute wd0 := (d0 - d6) + 5.25 * (d4 - d2)  */ \
         wd0 += const_5_25 * d4_sub_d2; \
         /*  Compute wd7 := (d7 - d1) + 5.25 * (d3 - d5)  */ \
         wd7 += const_5_25 * d3_sub_d5; \
 \
         const psimd_f32 const_4_25 = psimd_splat_f32(4.25f); \
         /*  Compute  */ \
         /*    wd1 := (d6 + d2) - 4.25 * d4  */ \
         /*    wd2 := (d1 + d5) - 4.25 * d3  */ \
         wd1 -= const_4_25 * d4; \
         wd2 -= const_4_25 * d3; \
 \
         const psimd_f32 const_1_25 = psimd_splat_f32(1.25f); \
         /*  Compute  */ \
         /*    wd3 := (d6 + 0.25 * d2) - 1.25 * d4  */ \
         /*    wd4 := (d5 + 0.25 * d1) - 1.25 * d3  */ \
         /*    wd6 := (d1 + 0.25 * d5) - 1.25 * d3  */ \
         /*    wd5 := (d6 - 5.0 * d4) + 4.0 * d2  */ \
         wd3 -= const_1_25 * d4; \
         const psimd_f32 d3_times_1_25 = d3 * const_1_25; \
         wd5 += psimd_splat_f32(4.0f) * d2; \
         wd4 -= d3_times_1_25; \
         wd6 -= d3_times_1_25; \
 \
         const psimd_f32 const_2 = psimd_splat_f32(2.0f); \
         wd4 *= const_2; \
         wd6 *= const_2; \
         psimd_f32_store( o0, wd0 ); \
         psimd_f32_store( o1, wd1 + wd2 ); \
         psimd_f32_store( o2, wd1 - wd2 ); \
         psimd_f32_store( o3, wd3 + wd4 ); \
         psimd_f32_store( o4, wd3 - wd4 ); \
         psimd_f32_store( o5, wd5 + wd6 ); \
         psimd_f32_store( o6, wd5 - wd6 ); \
         psimd_f32_store( o7, wd7 ); \
      }


      /*
       * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
       * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
       * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
       * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
       * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
       * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
       */
      #define TRANS_WINO_INV(i0, i1, i2, i3, i4, i5, i6, i7, \
                             o0, o1, o2, o3, o4, o5 ) { \
 \
      psimd_f32 m0(i0); \
      psimd_f32 m1(i1); \
      psimd_f32 m2(i2); \
      psimd_f32 m3(i3); \
      psimd_f32 m4(i4); \
      psimd_f32 m5(i5); \
      psimd_f32 m6(i6); \
      psimd_f32 m7(i7); \
 \
      const psimd_f32 m1_add_m2 = m1 + m2; \
      const psimd_f32 m1_sub_m2 = m1 - m2; \
      const psimd_f32 m3_add_m4 = m3 + m4; \
      const psimd_f32 m3_sub_m4 = m3 - m4; \
      const psimd_f32 m5_add_m6 = m5 + m6; \
      const psimd_f32 m5_sub_m6 = m5 - m6; \
 \
      psimd_f32 s0 = m0 + m1_add_m2; \
      psimd_f32 s5 = m7 + m1_sub_m2; \
 \
      const psimd_f32 const_16 = psimd_splat_f32(16.0f); \
      psimd_f32 s1 = m1_sub_m2 + const_16 * m5_sub_m6; \
      psimd_f32 s4 = m1_add_m2 + const_16 * m3_add_m4; \
 \
      const psimd_f32 const_8 = psimd_splat_f32(8.0f); \
      psimd_f32 s2 = m1_add_m2 + const_8 * m5_add_m6; \
      psimd_f32 s3 = m1_sub_m2 + const_8 * m3_sub_m4; \
 \
      const psimd_f32 const_32 = psimd_splat_f32(32.0f); \
      s0 += const_32 * m5_add_m6; \
      s5 += const_32 * m3_sub_m4; \
 \
      s0 += m3_add_m4; \
      s5 += m5_sub_m6; \
 \
      const psimd_f32 const_2 = psimd_splat_f32(2.0f); \
      s1 += m3_sub_m4 * const_2; \
      s4 += m5_add_m6 * const_2; \
 \
      const psimd_f32 const_4 = psimd_splat_f32(4.0f); \
      s2 += m3_add_m4 * const_4; \
      s3 += m5_sub_m6 * const_4; \
\
      psimd_f32_store( o0, s0 ); \
      psimd_f32_store( o1, s1 ); \
      psimd_f32_store( o2, s2 ); \
      psimd_f32_store( o3, s3 ); \
      psimd_f32_store( o4, s4 ); \
      psimd_f32_store( o5, s5 ); \
   }


      /*
        src is HWC format, 8 x 8 x inputs float32

        dest is HWC format wanting the central 6 x 6 x outputs float32
      */

      // Loop over source channels, four at a time
      // Src is in linear input components (features) per pixel, like:
      //  src = 
      //     i0 i1 i2 i3 i4 i5 i6 i7 i8 ... i0 i1 i2 i3 i4 i5 i6 i7 i8 ... i0 i1 i2 i3 i4 i5 i6 i7 i8 ...
      //
      //  source stride (pixel width) to get from one set of 4 inputs to the next
      int ss = inputs;
      // Transform into input coefficients per pixel
      //    c0 c1 c2 c3 c4 c5 c6 c7 c8 ... c0 c1 c2 c3 c4 c5 c6 c7 c8 ... c0 c1 c2 c3 c4 c5 c6 c7 c8 ...
      //
      // Do this by gathering 4 components at a time for all pixels, transforming to rows in scrach buffer,
      //  then transforming the cols of the scratch buffer over the top of the original source
      //
      for(int inputIdx=0; inputIdx<inputs; inputIdx+=4)
      {
         const float *src0 = src + inputIdx;
         float *out0 = scratch;

         // Transorm 4 channels to 4 channels of wonigrad coeffs
         // Tranform rows to buffer
         for(int i=0;i<8;i++)
         {
            TRANS_WINO( src0+0, src0+ss, src0+2*ss, src0+3*ss, src0+4*ss, src0+5*ss, src0+6*ss, src0+7*ss, \
                        out0+0, out0+4,  out0+8,    out0+12,   out0+16,   out0+20,   out0+24,   out0+28 );
            // Next input row
            src0 += inputs*8;
            // Next output row
            out0 += 32;
         }

         // Transform scratch buffer columns and store result over the original source
         const float *col0 = scratch;
         float *dest0 = src + inputIdx;
         int sy = inputs*8; // step to next input row
         for(int i=0;i<8;i++)
         {
            TRANS_WINO( col0+0, col0+32,  col0+64, col0+96, col0+128, col0+160, col0+192, col0+224,
                        dest0+0, dest0+sy, dest0+2*sy, dest0+3*sy, dest0+4*sy, dest0+5*sy, dest0+6*sy, dest0+7*sy );
            // Next column of 4 coeffs
            col0+=4;
            // Next destination row
            dest0 += inputs;
         }
      }

      /*
      const float *dbg = src;
      for(int o=0;o<4;o+=4)
      {
         printf("out %d\n",o);
         for(int px=0;px<64;px++)
         {
            printf(" t %d:", px);
            for(int i=0;i<inputs;i++)
               printf(" %g", *dbg++);
            printf("\n");
         }
      }
      */

      // Src values have now been converted to winograd coeffs in-situ
      //   src: c0 c1 c2 c3 c4 c5 c6 c7 c8 ... c0 c1 c2 c3 c4 c5 c6 c7 c8 ... c0 c1 c2 c3 c4 c5 c6 c7 c8 ...

      // For each output, o, we want transformed To = Sum_i w_oi ci
      //
      // Perform this sum for each pixel, 4 outputs at a time and store the result in the 8*8*4  scratch buffer
      //  then inverse-transform into the destination buffer.
      //


      float *w = (float *)transformWeights;
      int sixteens = inputs>>4;
      int fours = (inputs - (sixteens<<4))>>2;
      for(int o=0; o<outputs;o+=4)
      {
         float *out = scratch;
         const float *s = src;

         for(int pixel=0;pixel<64;pixel++)
         {
            float32x4_t sumO0 = Zero4f32;
            float32x4_t sumO1 = sumO0;
            float32x4_t sumO2 = sumO0;
            float32x4_t sumO3 = sumO0;


            for(int i=0;i<fours;i++)
            {
               float32x4_t src = Load4f32(s);
               s+=4;
               sumO0 = Add4f32( Mul4f32(src, Load4f32(w   ) ), sumO0 );
               sumO1 = Add4f32( Mul4f32(src, Load4f32(w+4 ) ), sumO1 );
               sumO2 = Add4f32( Mul4f32(src, Load4f32(w+8 ) ), sumO2 );
               sumO3 = Add4f32( Mul4f32(src, Load4f32(w+12) ), sumO3 );
               w+=16;
            }

            for(int i=0;i<sixteens;i++)
            {
               float32x4_t src0 = Load4f32(s);
               sumO0 = Add4f32( Mul4f32(src0, Load4f32(w   ) ), sumO0 );
               sumO1 = Add4f32( Mul4f32(src0, Load4f32(w+4 ) ), sumO1 );
               sumO2 = Add4f32( Mul4f32(src0, Load4f32(w+8 ) ), sumO2 );
               sumO3 = Add4f32( Mul4f32(src0, Load4f32(w+12) ), sumO3 );

               float32x4_t src1 = Load4f32(s+4);
               sumO0 = Add4f32( Mul4f32(src1, Load4f32(w+16) ), sumO0 );
               sumO1 = Add4f32( Mul4f32(src1, Load4f32(w+20) ), sumO1 );
               sumO2 = Add4f32( Mul4f32(src1, Load4f32(w+24) ), sumO2 );
               sumO3 = Add4f32( Mul4f32(src1, Load4f32(w+28) ), sumO3 );

               float32x4_t src2 = Load4f32(s+8);
               sumO0 = Add4f32( Mul4f32(src2, Load4f32(w+32) ), sumO0 );
               sumO1 = Add4f32( Mul4f32(src2, Load4f32(w+36) ), sumO1 );
               sumO2 = Add4f32( Mul4f32(src2, Load4f32(w+40) ), sumO2 );
               sumO3 = Add4f32( Mul4f32(src2, Load4f32(w+44) ), sumO3 );

               float32x4_t src3 = Load4f32(s+12);
               s+=16;
               sumO0 = Add4f32( Mul4f32(src3, Load4f32(w+48) ), sumO0 );
               sumO1 = Add4f32( Mul4f32(src3, Load4f32(w+52) ), sumO1 );
               sumO2 = Add4f32( Mul4f32(src3, Load4f32(w+56) ), sumO2 );
               sumO3 = Add4f32( Mul4f32(src3, Load4f32(w+60) ), sumO3 );
               w+=64;
            }


            SumRows4x4f32(sumO0, sumO1, sumO2, sumO3);
            Store4f32( out, sumO0 );
            out += 4;
         }

         // Inverse-winograd 4-coeffs into scratch buffer

         float *rowSrc = scratch;
         float *rowDest = scratch;
         // compact rows in-situ
         for(int i=0;i<8;i++)
         {
            TRANS_WINO_INV(rowSrc+0, rowSrc+4,  rowSrc+8,    rowSrc+12,   rowSrc+16,   rowSrc+20,   rowSrc+24,   rowSrc+28, \
                           rowDest+0, rowDest+4,  rowDest+8,    rowDest+12,   rowDest+16,   rowDest+20 );
            rowSrc+= 32;
            rowDest+=24;
         }

         // compact cols 
         float *c = scratch;
         for(int i=0;i<xCount;i++)
         {
            TRANS_WINO_INV(c, c+24, c+48, c+72, c+96, c+120, c+144, c+168, \
                           c, c+24, c+48, c+72, c+96, c+120 );
            c+=4;
         }

         // Activate and bias from scratch buffer to output
         float32x4_t zero = Zero4f32;
         float32x4_t biasVal = Load4f32(biasPtr + o);

         float *unactive = scratch;
         float *d = dest + o;

         if (activation==actRelu)
         {
            for(int oy=0; oy<yCount; oy++)
            {
               for(int ox=0; ox<xCount;ox++)
                  Store4f32( d + ox*outputs, Max4f32( Add4f32( Load4f32( unactive+ox*4 ), biasVal), zero ) );
               unactive += 24;
               d += outputs*destW;
            }
         }
         else if (activation==actLeaky)
         {
            float32x4_t alpha = Const4f32(0.1f);
            for(int oy=0; oy<yCount; oy++)
            {
               for(int ox=0; ox<xCount;ox++)
               {
                  float32x4_t val = Add4f32( Load4f32( unactive+ox*4 ), biasVal);
                  Store4f32( d + ox*outputs, Max4f32( val, Mul4f32(val,alpha) ) );
               }
               unactive += 24;
               d += outputs*destW;
            }
         }
         else
         {
            for(int oy=0; oy<yCount; oy++)
            {
               for(int ox=0; ox<xCount;ox++)
                  Store4f32( d + ox*outputs, Add4f32( Load4f32( unactive+ox*4 ), biasVal ) );
               unactive += 24;
               d += outputs*destW;
            }
         }
      }
   }

   void runThread(int threadId)
   {
      const int *srcStride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];


      int tilesX = (srcW + 5)/6;
      int tilesY = (srcH + 5)/6;
      int tileCount = tilesX * tilesY;

      const float *sIn = (float *)src0->cpuRead();
      float *sOut = (float *)destTensor->cpuRead();

      while(true)
      {
         int tileId = getNextJob();
         if (tileId>=tileCount)
            break;

         int ty = tileId/tilesX;
         int tx = tileId-ty*tilesX;

         float *buf = srcBuffers[threadId];
         int outY = ty*6;
         int outX = tx*6;
         int sy0 = outY - 1;
         int sy1 = std::min(sy0+8,srcH);
         int sx0 = outX - 1;
         int sxEnd = sx0+8;
         int sx1 = std::min(sxEnd,srcW);
         int syEnd = sy0+8;

         memset(buf,0xff,8*8*inputs*sizeof(float));

         if (ty==0)
         {
            memset(buf,0,inputs*8*sizeof(float)); 
            sy0++;
            buf += inputs*8;
         }

         for(int y=sy0;y<sy1;y++)
         {
            if (sx0<0)
            {
               memset(buf,0,inputs*sizeof(float)); 
               memcpy(buf+inputs, sIn + (y*srcW)*inputs,sx1*inputs*sizeof(float)); 
               buf += inputs*(sx1+1);
            }
            else
            {
               memcpy(buf, sIn + (y*srcW + sx0)*inputs,(sx1-sx0)*inputs*sizeof(float)); 
               buf += inputs*(sx1-sx0);
            }
            if (sx1 < sxEnd)
            {
               memset(buf, 0, (sxEnd-sx1)*inputs*sizeof(float));
               buf += (sxEnd-sx1)*inputs;
            }
         }
         if (sy1<syEnd)
         {
            memset(buf, 0, (syEnd-sy1)*inputs*8*sizeof(float));
         }

         /*
         const float *s = srcBuffers[threadId];
         for(int y=0;y<8;y++)
         {
            printf("%d ]",y);
            for(int x=0;x<8;x++)
            {
               printf("(%g,%g,%g,%g) ", s[0],s[1],s[2],s[3]);
               s+=4;
            }
            printf("\n");
         }
         */

         runTile( srcBuffers[threadId], sOut + (outY*destW + outX)*outputs,
                    std::min(outX+6,destW)-outX, std::min(outY+6,destH)-outY,
                    scratchBuffer[threadId]);
      }
   }


};
#endif





Layer *Layer::createConv2D(int strideY, int strideX,
                    Activation activation, Padding padding,
                    Tensor *weights, Tensor *pweights, Tensor *bias,
                    bool inAllowTransform)
{
   CShape filter = weights->shape;

   #ifdef NUMERIX_WINOGRAD
   if (inAllowTransform && filter.size()==4 && (filter[0]&3)==0 && filter[1]==3 && filter[2]==3 && (filter[3]&3)==0 &&
       pweights==0 && strideX==1 &&  strideY==1)
      return new Conv2DWinograd(strideY, strideX, activation, padding, weights, bias);
   #endif

   return new Conv2D(strideY, strideX, activation, padding, weights, pweights, bias);
}

} // end namespace numerix
