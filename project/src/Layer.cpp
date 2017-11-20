#include <Tensor.h>
#include <NxThread.h>

using namespace numerix;


float *Layer::allocFloats(int count,bool inZero)
{
   unsigned char *buffer = Tensor::allocData( count*sizeof(float) );
   if (inZero)
      memset(buffer, 0, count*sizeof(float));
   buffers.push_back(buffer);
   return (float *)buffer;
}

Layer::~Layer()
{
   for(int i=0;i<buffers.size();i++)
      Tensor::freeData( buffers[i] );
}

Tensor *Layer::makeOutput(Tensor *inBuffer, int inW, int inH, int inChannels, int inType)
{
   bool match = false;
   if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==inType)
   {
      CShape &s = inBuffer->shape;
      if (s[0]==inW && s[1]==inH && s[2]==inChannels)
         return inBuffer;
   }

   Shape s(3);
   s[0] = inW;
   s[1] = inH;
   s[2] = inChannels;
   return new Tensor( inType, s );
}


static void SRunThreaded(int inThreadId, void *thiz)
{
   ((Layer *)thiz)->runThread(inThreadId);
}

int Layer::getNextJob()
{
   return HxAtomicInc(&jobId);
}

void Layer::runThreaded()
{
   jobId = 0;
   //runThread(0);
   RunWorkerTask( SRunThreaded, this );
}
