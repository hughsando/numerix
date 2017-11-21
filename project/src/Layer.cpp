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
