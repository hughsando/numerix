#ifndef FAKE_PTHREAD_H
#define FAKE_PTHREAD_H

typedef bool pthread_once_t;

#define PTHREAD_ONCE_INIT false

inline void pthread_once(pthread_once_t *isInit, void (*initFunc)() )
{
   if (!*isInit)
   {
      *isInit = true;
      initFunc();
   }
}

#endif

