#ifndef OPS_H
#define OPS_H

#include "Tensor.h"

 #include <xmmintrin.h>
 #include <emmintrin.h>
// SSE4 ?
 #include <smmintrin.h>


#define ALLOW_SIMD

namespace numerix
{
typedef __m128 float32x4_t;
#define Load4f32(ptr) _mm_load_ps(ptr)
#define Add4f32(a,b) _mm_add_ps(a,b)
#define Mul4f32(a,b) _mm_mul_ps(a,b)
#define Zero4f32 _mm_castsi128_ps(_mm_setzero_si128())

inline float Accumulate4f32(float32x4_t v) {
     // v : abcd
    float32x4_t shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
     // shuf : badc
    float32x4_t sums   = _mm_add_ps(v, shuf);
     // shuf : (b+a) (b+a) (c+d) (c+d)
    shuf          = _mm_movehl_ps(shuf, sums);
    //  (c+d) x x x
    sums          = _mm_add_ss(sums, shuf);
    //  (a+b)+(c+d) x x x
    return    _mm_cvtss_f32(sums);
}



inline float dot(float s0,const float *w, const float *s, int n, Activation activation)
{
   float sum = s0;

   int eights = n>>3;
   if (eights)
   {
      #ifdef ALLOW_SIMD
      if (true)
      {
         float32x4_t sum4;

         float32x4_t srcA = Load4f32(s);
         float32x4_t srcB = Load4f32(s+4);
         s+=8;
         float32x4_t weiA = Load4f32(w);
         sum4 = Mul4f32(srcA,weiA);
         float32x4_t weiB = Load4f32(w+4);
         w+=8;

         int sixteens = (eights-1)>>1;
         for(int i=0;i<sixteens;i++)
         {
            srcA = Load4f32(s);
            sum4 =  Add4f32( sum4, Mul4f32(srcB, weiB) );
            srcB = Load4f32(s+4);
            s+=8;
            weiA = Load4f32(w);
            sum4 =  Add4f32( sum4,  Mul4f32(srcA, weiA) );
            weiB = Load4f32(w+4);
            w+=8;

            srcA = Load4f32(s);
            sum4 =  Add4f32( sum4, Mul4f32(srcB, weiB) );
            srcB = Load4f32(s+4);
            s+=8;
            weiA = Load4f32(w);
            sum4 =  Add4f32( sum4,  Mul4f32(srcA, weiA) );
            weiB = Load4f32(w+4);
            w+=8;
         }
         eights -= sixteens<<1;


         for(int i=1;i<eights;i++)
         {
            srcA = Load4f32(s);
            sum4 =  Add4f32( sum4, Mul4f32(srcB, weiB) );
            srcB = Load4f32(s+4);
            s+=8;
            weiA = Load4f32(w);
            sum4 =  Add4f32( sum4,  Mul4f32(srcA, weiA) );
            weiB = Load4f32(w+4);
            w+=8;
         }
         sum4 =  Add4f32( sum4, Mul4f32(srcB, weiB) );
         sum +=  Accumulate4f32(sum4);
      }
      else
      #endif
      {
         for(int i=0;i<eights;i++)
         {
            sum += s[0] * w[0] +
                   s[1] * w[1] +
                   s[2] * w[2] +
                   s[3] * w[3] +
                   s[4] * w[4] +
                   s[5] * w[5] +
                   s[6] * w[6] +
                   s[7] * w[7];
            s+=8;
            w+=8;
         }
      }
   }

   if (n&0x4)
   {
      sum += s[0] * w[0] +
             s[1] * w[1] +
             s[2] * w[2] +
             s[3] * w[3];
      s+=4;
      w+=4;
   }
   if (n&0x2)
   {
      sum += s[0] * w[0] +
             s[1] * w[1];
      s+=2;
      w+=2;
   }
   if (n&0x1)
      sum += s[0] * w[0];

   if (activation==actRelu && sum<0)
      sum = 0;
   else if (activation==actSigmoid)
      sum = 1.0 / (1.0 + exp(-sum));

   return sum;
}

float dotSkip(const float *w, const float *s, int n, int inSkip)
{
   float sum = 0;
   for(int i=0;i<n;i++)
   {
      sum += w[i]*(*s);
      s+=inSkip;
   }
   return sum;
}

} // end namespace numerix


#endif
