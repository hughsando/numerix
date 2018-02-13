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
protected:
   volatile int jobId;
   std::vector<unsigned char *> buffers;
   double runTime;
   double runStart;
   int    runCount;

public:
   typedef unsigned char u8;
   static bool accurateTimes;
   static bool openCLTimingEvents;

   Layer();


   static Layer *createConv2D(int inStrideY, int inStrideX, bool inIsDeconvolution,
                              Activation activation, Padding padding,
                              Tensor *weights, Tensor *pweights, Tensor *bias,
                              bool inAllowTransform);

   static Layer *createMaxPool(int inSizeX, int inSizeY,
                               int inStrideY, int inStrideX,
                               Padding padding);

   static Layer *createConcat();

   static Layer *createGlobalPool();

   static Layer *createSoftmax();

   static Layer *createReorg(int inStride);

   static Layer *createYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh);

   static Layer *createMovidius(const std::string &inDeviceName, const unsigned char *graphData, size_t dataLength, CShape outputSize);


   virtual ~Layer();

   virtual void setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars) { }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer) { return 0; }
   virtual Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer) { return 0; }

   virtual void runThread(int inThreadId) { }

   virtual void getBoxes(Boxes &outBoxes) { }

   virtual void setPadInput() { }

   virtual void setActivation(Activation inActivation) { }

   virtual double getRunTime();

   int getNextJob();
   void startRun();
   void endRun();

   float *allocFloats(int count,bool inZero=false);
   void   releaseFloats();


   void runThreaded(bool debug=false);
};







class Conv2DBase : public Layer
{
protected:
   int        strideX;
   int        strideY;
   int        filterY;
   int        filterX;
   int        inputs;
   int        outputs;
   int        diSize;
   bool       padInputsWithZero;
   bool       isDeconvolution;

   int        srcW;
   int        srcH;
   int        destW;
   int        destH;
   int        padOx;
   int        padOy;


   Activation activation;
   Padding    padding;
   Tensor     *weightsOriginal;
   Tensor     *weights;
   Tensor     *pweights;
   Tensor     *bias;

   bool       is1x1Aligned;
   bool       is1x1;



public:
   Conv2DBase(int inStrideY, int inStrideX, bool isDeconvolution,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inPWeights, Tensor *inBias);

   ~Conv2DBase();

   void setPadInput() { padInputsWithZero = true; }

   virtual void rebuildWeights() { };

   void setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars);

   void setActivation(Activation inActivation) { activation=inActivation; }

   void reduceInputs(int inCount);

   Tensor *run(Tensor *inSrc0, Tensor *inBuffer);

   virtual void doRun(Tensor *input, Tensor *output) = 0;
};

























} // end namespace numerix

#endif
