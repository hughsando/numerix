#ifndef OPS_H
#define OPS_H

#include "Tensor.h"
#include <math.h>

#if defined(__arm__)
   #define NUMERIX_NEON
#endif

#ifdef NUMERIX_SIMD
  #ifdef NUMERIX_NEON
     #include <arm_neon.h>
  #else
     #include <xmmintrin.h>
     #include <emmintrin.h>
     // SSE4 ?
     #include <smmintrin.h>
  #endif
#endif

namespace numerix
{

#if defined(NUMERIX_SIMD) && defined(NUMERIX_NEON)

typedef float AlignedFloat __attribute__((aligned(sizeof(float32x4_t))));

#define Load4f32(ptr) vld1q_f32( (const AlignedFloat*) (ptr) )
#define Add4f32(a,b) vaddq_f32(a,b)
#define Sub4f32(a,b) vsubq_f32(a,b)
#define Mul4f32(a,b) vmulq_f32(a,b)
#define Zero4f32 vdupq_n_f32(0.0f)
#define Const4f32(c) vdupq_n_f32(c)
#define Max4f32(a,b) vmaxq_f32(a,b)
#define Store4f32(ptr, value)  vst1q_f32( (AlignedFloat *) (ptr) , (value) )
#define LaneF32(val, lane)  vgetq_lane_f32( val , lane )

inline float Accumulate4f32(float32x4_t v)
{
   // v : a0 a1 a2 a3
   float32x2_t sumA = vpadd_f32(vget_low_f32(v),vget_high_f32(v));
   //  ->  a0+a1 a2+a3
   float32x2_t sumB = vpadd_f32(sumA,sumA);
   return vget_lane_f32(sumB,0);
}

   // Sum0 : a0 a1 a2 a3
   // Sum1 : b0 b1 b2 b3
   //  ->  a0+a1 a2+a3
   //  ->  b0+b1 b2+b3
   //  ->  a0+a1+a2+a3  b0+b1+b2+b3
   //  ->  c0+c1+c2+c3  c0+c1+c2+c3
#define SumRows4x4f32(sum0, sum1, sum2, sum3) \
   float32x2_t sumA = vpadd_f32(vget_low_f32(sum0),vget_high_f32(sum0)); \
   float32x2_t sumB = vpadd_f32(vget_low_f32(sum1),vget_high_f32(sum1)); \
   float32x2_t sAB = vpadd_f32(sumA,sumB); \
 \
   float32x2_t sumC = vpadd_f32(vget_low_f32(sum2),vget_high_f32(sum2)); \
   float32x2_t sumD = vpadd_f32(vget_low_f32(sum3),vget_high_f32(sum3)); \
   float32x2_t sCD = vpadd_f32(sumC,sumD); \
 \
   sum0 = vcombine_f32(sAB, sCD);



#elif defined(NUMERIX_SIMD)

typedef __m128 float32x4_t;
#define Load4f32(ptr) _mm_load_ps(ptr)
#define Add4f32(a,b) _mm_add_ps(a,b)
#define Sub4f32(a,b) _mm_sub_ps(a,b)
#define Mul4f32(a,b) _mm_mul_ps(a,b)
#define Zero4f32 _mm_castsi128_ps(_mm_setzero_si128())
#define Const4f32(c) _mm_set1_ps(c)
#define Max4f32(a,b) _mm_max_ps(a,b)
#define Store4f32(ptr, value)  _mm_store_ps(ptr, value)
#define LaneF32(a, lane)  (a).m128_f32[lane]

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

#define SumRows4x4f32(a, b, c, d) \
    /* [a0+a2 c0+c2 a1+a3 c1+c3 */ \
    a = _mm_add_ps(_mm_unpacklo_ps(a,c),_mm_unpackhi_ps(a,c)); \
    /* [b0+b2 d0+d2 b1+b3 d1+d3 */ \
    b = _mm_add_ps(_mm_unpacklo_ps(b,d),_mm_unpackhi_ps(b,d)); \
    /* [a0+a2 b0+b2 c0+c2 d0+d2]+ \
       [a1+a3 b1+b3 c1+c3 d1+d3] */ \
    a = _mm_add_ps(_mm_unpacklo_ps(a,b),_mm_unpackhi_ps(a,b)); \

/*
#define SumRows4x4f32(sum0, sum1, sum2, sum3) \
      _MM_TRANSPOSE4_PS(sum0, sum1, sum2, sum3) \
      sum0 = Add4f32(sum0,sum1); \
      sum2 = Add4f32(sum2,sum3); \
      sum0 = Add4f32(sum0,sum2);
*/
#endif

#if defined(NUMERIX_SIMD)
struct psimd_f32
{
   float32x4_t val;

   inline psimd_f32() { }
   inline psimd_f32(const psimd_f32 &inVal) : val(inVal.val) {  }
   inline psimd_f32(const float32x4_t &inVal) : val(inVal) { }
   inline psimd_f32(const float *inPtr) : val(Load4f32(inPtr)) { }


   inline psimd_f32 operator+(const psimd_f32 &o) const { return Add4f32(val,o.val); }
   inline psimd_f32 operator*(const psimd_f32 &o) const { return Mul4f32(val,o.val); }
   inline psimd_f32 operator-(const psimd_f32 &o) const { return Sub4f32(val,o.val); }
   inline psimd_f32 &operator+=(const psimd_f32 &o) { val = Add4f32(val,o.val); return *this;  }
   inline psimd_f32 &operator*=(const psimd_f32 &o) { val = Mul4f32(val,o.val); return *this;  }
   inline psimd_f32 &operator-=(const psimd_f32 &o) { val = Sub4f32(val,o.val); return *this;  }
   //inline psimd_f32 operator-() const { return _mm_xor_ps(val, Const4f32(-0.f));  }
   
   inline float operator[](int lane) { return LaneF32(val, lane); }
};
inline psimd_f32 psimd_splat_f32(float c) { return Const4f32(c); }
inline void psimd_f32_store(float *outPtr, const psimd_f32 &inVal) { Store4f32(outPtr, inVal.val); }

#endif

/*
 Here we attempt to re-use the same 4 source channels with 4 consecutive outputs.
 The weights have been pre-arranged to have 4 outputs grouped into blocks of 4, like:
  W :   A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 A4 A5 A6 A7 B4 .....
Src :   S0 S1 S2 S3  S4 S5 S6 ....

We want outputA = S0*A0 + S1*A1 + S2*A2 + S3*A3 + S4*A4 + ...
        outputB = S0*B0 + S1*B1 + S2*B2 + S3*B3 + S4*B4 + ...
        outputC = S0*C0 + S1*C1 + S2*C2 + S3*C3 + S4*C4 + ...
        outputD = S0*D0 + S1*D1 + S2*D2 + S3*D3 + S4*D4 + ...

The inputs have been padded to be multiples of 4, and n is the input count/4

*/

inline void dot4Interlaced(float *dest, const float *alignedBias, const float *wABCD, const float *inSrc, int n, Activation activation)
{
   #ifdef NUMERIX_SIMD
   if (true)
   {
      int eights = n>>1;

      float32x4_t src;
      float32x4_t sum0;
      float32x4_t sum1;
      float32x4_t sum2;
      float32x4_t sum3;
      float32x4_t wa;

      const float *w = wABCD;
      const float *srcPtr = inSrc;

      #define AccumulateF32x4 \
         src = Load4f32(srcPtr); \
         srcPtr += 4; \
         wa = Load4f32(w); \
         wa = Mul4f32(src,wa); \
         sum0 = Add4f32(sum0,wa); \
    \
         wa = Load4f32(w+4); \
         wa = Mul4f32(src,wa); \
         sum1 = Add4f32(sum1,wa); \
   \
         wa = Load4f32(w+8); \
         wa = Mul4f32(src,wa); \
         sum2 = Add4f32(sum2,wa); \
    \
         wa = Load4f32(w+12); \
         w+=16; \
         wa = Mul4f32(src,wa); \
         sum3 = Add4f32(sum3,wa);


      // First one - no 'add', just 'mul'
      src = Load4f32(srcPtr);
      srcPtr += 4;
      wa = Load4f32(w);
      sum0 = Mul4f32(src,wa);

      wa = Load4f32(w+4);
      sum1 = Mul4f32(src,wa);

      wa = Load4f32(w+8);
      sum2 = Mul4f32(src,wa);

      wa = Load4f32(w+12);
      w+=16;
      sum3 = Mul4f32(src,wa);

      int fours = n-1;

      if (fours&1)
      {
         AccumulateF32x4;
      }

      if (fours&2)
      {
         AccumulateF32x4;
         AccumulateF32x4;
      }

      int sixteens = fours>>2;
      for(int i=0;i<sixteens;i++)
      {
         AccumulateF32x4;
         AccumulateF32x4;
         AccumulateF32x4;
         AccumulateF32x4;
      }

      SumRows4x4f32(sum0, sum1, sum2, sum3)

      float32x4_t bias = Load4f32(alignedBias);


      sum0 = Add4f32(sum0,bias);
      if (activation==actRelu)
         sum0 = Max4f32(sum0, Zero4f32);
      else if (activation==actLeaky)
         sum0 = Max4f32( sum0, Mul4f32(sum0, Const4f32(0.1f)) );
      Store4f32(dest, sum0);
   }
   else
   #endif
   {
      const float *src = inSrc;
      float sum0 = alignedBias[0];
      float sum1 = alignedBias[1];
      float sum2 = alignedBias[2];
      float sum3 = alignedBias[3];
      for(int i=0;i<n;i++)
      {
         sum0 += src[0]*wABCD[0] + src[1]*wABCD[1] + src[2]*wABCD[2] + src[3]*wABCD[3];
         wABCD += 4;
         sum1 += src[0]*wABCD[0] + src[1]*wABCD[1] + src[2]*wABCD[2] + src[3]*wABCD[3];
         wABCD += 4;
         sum2 += src[0]*wABCD[0] + src[1]*wABCD[1] + src[2]*wABCD[2] + src[3]*wABCD[3];
         wABCD += 4;
         sum3 += src[0]*wABCD[0] + src[1]*wABCD[1] + src[2]*wABCD[2] + src[3]*wABCD[3];
         wABCD += 4;
         src+=4;
      }

      if (activation==actRelu)
      {
         dest[0] = sum0 < 0 ? 0 : sum0;
         dest[1] = sum1 < 0 ? 0 : sum1;
         dest[2] = sum2 < 0 ? 0 : sum2;
         dest[3] = sum3 < 0 ? 0 : sum3;
      }
      else if (activation==actLeaky)
      {
         dest[0] = sum0 < 0 ? sum0*0.1f : sum0;
         dest[1] = sum1 < 0 ? sum1*0.1f : sum1;
         dest[2] = sum2 < 0 ? sum2*0.1f : sum2;
         dest[3] = sum3 < 0 ? sum3*0.1f : sum3;
      }
      else
      {
         dest[0] = sum0;
         dest[1] = sum1;
         dest[2] = sum2;
         dest[3] = sum3;
      }
   }
}



inline float dot(float s0,const float *w, const float *s, int n, Activation activation)
{
   float sum = s0;

   int eights = n>>3;
   if (eights)
   {
      #ifdef NUMERIX_SIMD
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
   else if (activation==actLeaky)
   {
      if (sum<0)
         sum*= 0.1f;
   }
   else if (activation==actSigmoid)
      sum = 1.0 / (1.0 + exp(-sum));

   return sum;
}

inline float dotSkip(const float *w, const float *s, int n, int inSkip)
{
   float sum = 0;
   for(int i=0;i<n;i++)
   {
      sum += w[i]*(*s);
      s+=inSkip;
   }
   return sum;
}



void floattofp16(unsigned char *dst, const float *src, unsigned nelem);
void fp16tofloat(float *dst, const unsigned char *src, unsigned nelem);


} // end namespace numerix


#endif
