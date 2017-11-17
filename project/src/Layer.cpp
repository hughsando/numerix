#include <Tensor.h>
#include <NxThread.h>

using namespace numerix;


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
