#include <Tensor.h>
#include <Layer.h>
#include <NxThread.h>
#include <memory.h>


namespace numerix
{

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
   releaseFloats();
}


void Layer::releaseFloats()
{
   for(int i=0;i<buffers.size();i++)
      Tensor::freeData( buffers[i] );
   buffers.resize(0);
}

static void SRunThreaded(int inThreadId, void *thiz)
{
   ((Layer *)thiz)->runThread(inThreadId);
}

int Layer::getNextJob()
{
   return HxAtomicInc(&jobId);
}

void Layer::runThreaded(bool inDebug)
{
   jobId = 0;
   if (inDebug)
      runThread(0);
   else
      RunWorkerTask( SRunThreaded, this );
}

} // namespace numerix
