#include <NxThread.h>

namespace numerix
{

ThreadId GetThreadId()
{
   #ifdef HX_WINDOWS
   return GetCurrentThreadId();
   #elif defined(EMSCRIPTEN)
   return 0;
   #else
   return pthread_self();
   #endif
}



static ThreadId sMainThread = 0;

void SetMainThread()
{
   sMainThread = GetThreadId();
}

bool IsMainThread()
{
   return sMainThread==GetThreadId();
}



volatile int gTaskId = 0;

int sWorkerCount = 0;
volatile int gActiveThreads = 0;

#ifdef __arm__
   #define MAX_NX_THREADS 4
#else
   #define MAX_NX_THREADS 8
#endif

#ifdef NX_PTHREADS
static NxMutex sThreadPoolLock;
typedef pthread_cond_t ThreadPoolSignal;
inline void WaitThreadLocked(ThreadPoolSignal &ioSignal)
{
   pthread_cond_wait(&ioSignal, &sThreadPoolLock.mMutex);
}
#else
typedef HxSemaphore ThreadPoolSignal;
#endif

bool sThreadActive[MAX_NX_THREADS];
ThreadPoolSignal sThreadWake[MAX_NX_THREADS];
ThreadPoolSignal sThreadJobDone;

static WorkerFunc sThreadJob = 0;
static void *sThreadData = 0;

static THREAD_FUNC_TYPE SThreadLoop( void *inInfo )
{
   int threadId = (int)(size_t)inInfo;
   while(true)
   {
      // Wait ....
      #ifdef NX_PTHREADS
      {
         NxAutoMutex l(sThreadPoolLock);
         while( !sThreadActive[threadId] )
            WaitThreadLocked(sThreadWake[threadId]);
      }
      #else
      while( !sThreadActive[threadId] )
         sThreadWake[threadId].Wait();
      #endif

      // Run
      sThreadJob( threadId, sThreadData );

      // Signal
      sThreadActive[threadId] = false;
      if (HxAtomicDec(&gActiveThreads)==1)
      {
         #ifdef NX_PTHREADS
         NxAutoMutex lock(sThreadPoolLock);
         pthread_cond_signal(&sThreadJobDone);
         #else
         sThreadJobDone.Set();
         #endif
      }
   }
   THREAD_FUNC_RET;
}



void initWorkers()
{
   sWorkerCount = MAX_NX_THREADS;

   #ifdef NX_PTHREADS
   pthread_cond_init(&sThreadJobDone,0);
   #endif

   for(int t=0;t<sWorkerCount;t++)
   {
      sThreadActive[t] = false;
      #ifdef NX_PTHREADS
      pthread_cond_init(&sThreadWake[t],0);
      pthread_t result = 0;
      int created = pthread_create(&result,0,SThreadLoop, (void *)(size_t)(int)t);
      bool ok = created==0;
      #else
      bool ok = HxCreateDetachedThread(SThreadLoop, (void *)(size_t)(int)t);
      #endif
   }
}

int GetWorkerCount()
{
   if (!sWorkerCount)
      initWorkers();
   return sWorkerCount;
}


extern "C" {
size_t pthreadpool_get_threads_count(struct pthreadpool *)
{
   return GetWorkerCount();
}
}



void RunWorkerTask( WorkerFunc inFunc, void *inData )
{
   gTaskId = 0;
   if (!sWorkerCount)
      initWorkers();

   gActiveThreads = sWorkerCount;
   sThreadJob = inFunc;
   sThreadData = inData;


   #ifdef NX_PTHREADS
   NxAutoMutex lock(sThreadPoolLock);
   #endif

   for(int t=0;t<sWorkerCount;t++)
   {
      sThreadActive[t] = true;
      #ifdef NX_PTHREADS
      pthread_cond_signal(&sThreadWake[t]);
      #else
      sThreadWake[t].Set();
      #endif
   }

   #ifdef NX_PTHREADS
   while(gActiveThreads)
      WaitThreadLocked(sThreadJobDone);
   #else
   while(gActiveThreads)
      sThreadJobDone.Wait();
   #endif

   sThreadJob = 0;
   sThreadData = 0;
}



} // end namespace numerix

