#include <Tensor.h>
#include <Ops.h>
#include <NxThread.h>
#include <stdexcept>
#include <algorithm>

using namespace numerix;

class YoloLayer : public Layer
{


   Tensor *dummy;
   std::vector<float> anchors;
   int boxCount;
   int classCount;
   float thresh;
   Boxes boxes;


   inline float logistic_activate(float f)
   {
      return 1.0 / (1.0 + exp(-f));
   }

   void softmax(const float *input, int n, float *output)
   {
       int i;
       float sum = 0;
       float largest = input[0];
       for(i = 1; i < n; ++i){
           if(input[i] > largest) largest = input[i];
       }
       for(i = 0; i < n; ++i){
           float e = exp(input[i] - largest);
           sum += e;
           output[i] = e;
       }
       for(i = 0; i < n; ++i){
           output[i] /= sum;
       }
   }

public:
   YoloLayer(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount) :
      anchors(inAnchors), boxCount(inBoxCount), classCount(inClassCount)
   {
      dummy = new Tensor(Float32,Shape1(1));
      dummy->zero(0,1);
      thresh = 0.24;
   }
   ~YoloLayer()
   {
      dummy->decRef();
   }

   Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      boxes.resize(0);
      int h = inSrc0->shape[0];
      int w = inSrc0->shape[1];
      int channels = inSrc0->shape[2];
      const float *src = (const float *)inSrc0->data;
      std::vector<float> softmaxBuf(classCount);

      int boxSize = 4 + 1 + classCount;
      for(int y=0;y<h;y++)
      {
         for(int x=0;x<w;x++)
         {
            const float *predictions = src + (y*w + x) * channels;
            for(int b=0;b<boxCount;b++)
            {
               const float *box = predictions;
               float scale = logistic_activate( predictions[4] );
               const float *classes = predictions + 5;

               softmax(classes,classCount,&softmaxBuf[0]);
               int foundClass = -1;
               float foundBest = thresh;
               for(int j=0;j<classCount;j++)
               {
                  float prob = scale * softmaxBuf[j];
                  if (prob>foundBest)
                  {
                     foundClass = j;
                     foundBest = prob;
                  }
               }

               if (foundClass>=0)
               {
                  BBox bbox;
                  bbox.classId = foundClass;
                  bbox.prob = foundBest;
                  bbox.x = (x + logistic_activate(box[0]))/w;
                  bbox.y = (y + logistic_activate(box[1]))/h;
                  bbox.w = exp(box[2]) * anchors[2*b] / w;
                  bbox.h = exp(box[3]) * anchors[2*b+1] / h;
                  boxes.push_back(bbox);
               }

               predictions += boxSize;
            }
         }
      }

      if (boxes.size()>1)
      {
         std::sort( boxes.begin(), boxes.end() );
         for(int i=1;i<boxes.size(); /* */ )
         {
            bool overlap = false;
            for(int j=0;j<i;j++)
               if (boxes[i].overlaps(boxes[j]))
               {
                  overlap = true;
                  break;
               }
            if (overlap)
               boxes.erase( boxes.begin() + i );
            else
               i++;
         }
      }

      return dummy;
   }

   
   void getBoxes(Boxes &outBoxes)
   {
      outBoxes = boxes;
   }


};



Layer *Layer::createYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount)
{
   return new YoloLayer(inAnchors, inBoxCount, inClassCount);
}

