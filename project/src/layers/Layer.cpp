#include <Tensor.h>
#include <Layer.h>
#include <NxThread.h>
#include <memory.h>


namespace numerix
{

bool Layer::accurateTimes = false;

float *Layer::allocFloats(int count,bool inZero)
{
   unsigned char *buffer = TensorData::allocCpuAligned( count*sizeof(float) );
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
      TensorData::freeCpuAligned( buffers[i] );
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


void SortBoxes(Boxes &ioBoxes)
{
   if (ioBoxes.size()>1)
   {
      std::sort( ioBoxes.begin(), ioBoxes.end() );
      for(int i=1;i<ioBoxes.size(); /* */ )
      {
         bool overlap = false;
         for(int j=0;j<i;j++)
            if (ioBoxes[i].overlaps(ioBoxes[j]))
            {
               overlap = true;
               break;
            }
         if (overlap)
            ioBoxes.erase( ioBoxes.begin() + i );
         else
            i++;
      }
   }
}



} // namespace numerix
