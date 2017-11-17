#ifndef OPS_H
#define OPS_H

#include "Tensor.h"

namespace numerix
{
   inline float dot(float s0,const float *w, const float *s, int n, Activation activation)
   {
      float sum = s0;
      for(int i=0;i<n;i++)
         sum += w[i]*s[i];

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
}


#endif
