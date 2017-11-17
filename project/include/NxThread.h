#ifndef NX_THREAD_H
#define NX_THREAD_H

#include <hx/Thread.h>

#ifndef HX_WINDOWS
#include <pthread.h>
#define NX_PTHREADS
#else
#include <windows.h>
#undef min
#undef max
#endif

namespace numerix
{

#ifndef HX_WINDOWS
typedef pthread_t ThreadId;
#else
typedef DWORD ThreadId;
#endif

ThreadId GetThreadId();
bool IsMainThread();
void SetMainThread();

typedef HxMutex NxMutex;

struct NxAutoMutex
{
   NxMutex &mutex;
   NxAutoMutex(NxMutex &inMutex) : mutex(inMutex)
   {
      mutex.Lock();
   }
   ~NxAutoMutex()
   {
     mutex.Unlock();
   }
};


extern int GetWorkerCount();
typedef void (*WorkerFunc)(int inThreadId, void *inData);
void RunWorkerTask( WorkerFunc inFunc, void *inData );


}

#endif

