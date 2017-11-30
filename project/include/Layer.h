#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "Tensor.h"

namespace numerix
{




#undef min
#undef max

struct BBox
{
   float x;
   float y;
   float w;
   float h;
   float prob;
   int   classId;

   inline bool operator<(const BBox &inRight) const
   {
      return prob > inRight.prob;
   }

   inline bool overlaps(const BBox &o) const
   {
      float intX = std::min(x+w, o.x+o.w) - std::max(x,o.x);
      if (intX<=0)
         return false;
      float intY = std::min(y+h, o.y+o.h) - std::max(y,o.y);
      if (intY<=0)
         return false;
      float intersect = intX*intY;
      float total = w*h + o.w*o.h - intersect;
      return (intersect > total * 0.5);
   }
};

typedef std::vector<BBox> Boxes;

void SortBoxes(Boxes &ioBoxes);



class Layer
{
   volatile int jobId;
   std::vector<unsigned char *> buffers;

public:
   typedef unsigned char u8;

   static Layer *createConv2D(int inStrideY, int inStrideX,
                              Activation activation, Padding padding,
                              Tensor *weights, Tensor *pweights, Tensor *bias);

   static Layer *createMaxPool(int inSizeX, int inSizeY,
                               int inStrideY, int inStrideX,
                               Padding padding);

   static Layer *createConcat();

   static Layer *createReorg(int inStride);

   static Layer *createYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh);


   virtual ~Layer();

   virtual void setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars) { }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer) { return 0; }
   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer) { return 0; }

   virtual void runThread(int inThreadId) { }

   virtual void getBoxes(Boxes &outBoxes) { }

   virtual void setPadInput() { }

   int getNextJob();

   float *allocFloats(int count,bool inZero=false);
   void   releaseFloats();


   void runThreaded(bool debug=false);
};


} // end namespace numerix

#endif
