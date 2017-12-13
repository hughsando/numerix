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
      TensorThrow("Conv2D - bias does not match output size");


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
      int padX = (destW - 1)*strideX + filterX - srcW;
      padOx = padX>>1;

      destH = (srcH+strideY-1)/strideY;
      int padY = (destH - 1)*strideY + filterY - srcH;
      padOy = padY>>1;
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
      src0 = input;
      destTensor = output;
      src0->cpuRead();
      destTensor->cpuWrite();

      runThreaded();
      src0 = 0;
      destTensor = 0;
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

         const float *src = (const float *)src0->cpuRead() + srcW*inputs*y;
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
            src+=inputs;
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
         int srcY = y;

         int dyMin = std::max(padOy-srcY,0);
         int dyMax = std::min(srcH+padOy-srcY,filterY);

         if (dyMin>0)
            memset(srcPtr,0,filterRow*dyMin);

         if (dyMax<filterY)
            memset(srcPtr+dyMax*filterW,0,filterRow*(filterY-dyMax) );

         for(int x=0;x<destW;x++)
         {
            int srcX = x;

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


#ifdef NUMERIX_WINOGRAD
class Conv2DWinograd : public Conv2DBase
{
   std::vector <float *> srcBuffers;
   std::vector <float *> srcTransBuffers;
   std::vector <float *> outputTransBuffers;

public:
   Conv2DWinograd(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inActivation, inPadding,  inWeights, 0, inBias)
   {
      rebuildWeights();
   }

   void rebuildWeights()
   {
      releaseFloats();

      srcBuffers.resize(0);
      srcTransBuffers.resize(0);
      outputTransBuffers.resize(0);


      int threads = GetWorkerCount();

      for(int i=0;i<threads;i++)
      {
         srcBuffers.push_back( allocFloats(8*8*inputs,false) );
         srcTransBuffers.push_back( allocFloats(8*8*4,false) );
         outputTransBuffers.push_back( allocFloats(8*8*outputs,false) );
      }
   }


   Tensor *src0;
   Tensor *destTensor;

   virtual void doRun(Tensor *input, Tensor *output)
   {
      src0 = input;
      destTensor = output;
      src0->cpuRead();
      destTensor->cpuWrite();

      //runThreaded();
      src0 = 0;
      destTensor = 0;
   }

   inline void runTile( const float *src, float *dest,
                       int xCount, int yCount, 
                       float *sBuf, float *oBuf)
   {
      /*
        src is HWC format, 8 x 8 x inputs float32

        dest is HWC format wanting the central 6 x 6 x outputs float32
      */

      // Loop over source channels, four at a time
      int ss = inputs;
      for(inputIdx=0; inputIdx<inputs; inputIdx+=4)
      {
         const float *s0 = src + inputIdx;
         float *s1 = sBuf;
         // Transorm 4 channels to 4 channels of wonigrad coeffs

         // Tranform rows to buffer
         for(int i=0;i<8;i++)
         {
            TRANS_WINO( s0+0, s0+ds, s0+2*ds, s0+3*ds, s0+4*ds, s0+5*ds, s0+6*ds, s0+7*ds, \
                        s1+0, s1+4,  s1+8,    s1+12,   s1+16,   s1+20,   s1+24,   s1+28 );
            s0 += inputs*8;
            s1 += 8*4;
         }
         // Transform buffer columns in-situ
         s1 = sBuf;
         for(int i=0;i<8;i++)
         {
            TRANS_WINO( s1+0, s1+32,  s1+64, s1+96, s1+128, s1+160, s1+192, s1+224,
                        s1+0, s1+32,  s1+64, s1+96, s1+128, s1+160, s1+192, s1+224 );
            s1++;
         }

         // Accumulate T(channel_i * weight_t) for each output an these 4 inputs
         float32x4_t *w = (float32x4_t)transformWeights;
         for(int o=0; o<outputs;o+=4)
         {
            float32x4_t *inCoeff = (float32x4_t *)sBuf;
            float32x4_t *outCh = (float32x4_t *)oBuf + o*8*8;
            // Accumulate 4 output channels
            if (inputIdx==0)
            {
               // just set
               for(int pixel=0;pixel<64;pixel++)
               {
                  float32x4_t coeff = Load4f32(inCoeff++);
                  float32x4_t wei   = Load4f32(w++);
                  Store4f32( outCh++, Mul4f32(coeff,wei) );
               }
            }
            else
            {
               // accumulate
               for(int pixel=0;pixel<64;pixel++)
               {
                  float32x4_t val = Load4f32(outCh);
                  float32x4_t coeff = Load4f32(inCoeff++);
                  float32x4_t wei   = Load4f32(w++);
                  Store4f32( outCh++, Add4f32(val, Mul4f32(coeff,wei)) );
               }
            }
         }
      }

      // Inverse-winograd output channel coeff-slabs
      for(int o=0; o<outputs;o+=4)
      {
         float *s0 = oBuf + o*8*8*4;
         float *s1 = sBuf;

         // compact rows...
         for(int i=0;i<8;i++)
         {
            TRANS_WINO_INV(s0+0, s0+4,  s0+8,    s0+12,   s0+16,   s0+20,   s0+24,   s0+28, \
                           s1+0, s1+4,  s1+8,    s1+12,   s1+16,   s1+20 );
            s0+= 32;
            s1+=24;
         }

         // compact cols...
         float *s1 = sBuf;
         for(int i=0;i<xCount;i++)
         {
            TRANS_WINO_INV(s1+24, s1+48, s1+72, s1+96, s1+120, s1+144, s1+168, s1+192, \
                           s1+24, s1+48, s1+72, s1+96, s1+120, s1+144 );
            s1+=4;
         }

         // Activate and bias ...
         float *src = s1;
         float32x4_t zero = Zero4f32();
         float32x4_t biasVal = Load4f32(bias + o);
         float *out = dest + o;
         for(int oy=0; oy<yCount; oy++)
         {
            for(int ox=0; ox<xCount;ox++)
               Store4f32( out + ox*4, Max4f32( Add4f32( Load4f32( src+ox*4 ), biasVal) ) );
            src += 24;
            out += outputs*destW;
         }
      }
   }

   void runThread(int threadId)
   {
      const float *b = bias ? (const float *)bias->cpuRead() : 0;
      const int *srcStride = &src0->strides[0];
      const int *destStride = &destTensor->strides[0];


      int tilesX = (srcW + 5)/6;
      int tilesY = (srcH + 5)/6;
      int tileCount = tilesX * tilesY;

      const float *sIn = (float *)src0->cpuRead();
      float *sOut = (float *)destTensor->cpuRead();

      while(true)
      {
         int tid = getNextJob();
         if (tid>=tileCount)
            break;

         int ty = tid/tilesX;
         int tx = tid-ty*tilesX;

         float *buf = srcBuffers[tid];
         int outY = ty*6;
         int outX = tx*6;
         int sy0 = outY - 1;
         int sy1 = std::min(sy0+8,srcH);
         int sx0 = outX - 1;
         int sxEnd = sx0+8;
         int sx1 = std::min(sxEnd,srcW);

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
               buf += inputs;
            }
            memcpy(buf, sIn + (y*srcW + sx0)*inputs,(sx1-sx0)*sizeof(float)); 
            buf += inputs*(sx1-sx0);
            if (sx0 < sxEnd)
            {
               memset(buf, 0, (sxEnd-sx0)*inputs*sizeof(float));
               buf += (sxEnd-sx0)*inputs;
            }
         }

         runTile( srcBuffers[tid], sOut + (outY*destW + outX)*outputs,
                    std::min(outX+6,destW)-outX, std::min(outY+6,destH)-outY,
                    srcTransBuffers[tid], outputTransBuffers[tid] );
      }
   }


};
#endif





Layer *Layer::createConv2D(int strideY, int strideX,
                    Activation activation, Padding padding,
                    Tensor *weights, Tensor *pweights, Tensor *bias)
{
   CShape filter = weights->shape;

   #ifdef NUMERIX_WINOGRAD
   if (filter.size()==4 && (filter[0]&3)==0 && filter[1]==3 && filter[2]==3 && (filter[3]&3)==0 &&
       pweights==0 )
      return new Conv2DWinograd(strideY, strideX, activation, padding, weights, bias);
   #endif

   return new Conv2D(strideY, strideX, activation, padding, weights, pweights, bias);
}

} // end namespace numerix
