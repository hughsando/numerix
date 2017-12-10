#include <dlfcn.h>
#include <stdio.h>
#include "mvnc.h"


extern "C" {


static void *libHandle = 0;
static bool tried = false;


void *LoadLibMvnc(const char *inName)
{
   void *dlopen(const char *filename, int flag);
   if (!libHandle && !tried)
   {
      const char *name = "libmvnc.so";
      tried = true;
      libHandle = dlopen(name, RTLD_GLOBAL|RTLD_NOW);
      //printf("Lib handle %s = %p\n", name, libHandle);
   }
   if (!libHandle)
      return 0;
   void *result = dlsym(libHandle, inName);
   //printf("%s -> %p\n", inName, result);
   return result;
}

#define DMVNC1(  NAME, A ) \
 mvncStatus NAME(A a) { \
    typedef mvncStatus (*f)(A); \
    static f func=0; \
    if (!func) func = (f)LoadLibMvnc(#NAME); \
    if (!func) { return MVNC_NO_DRIVER; } \
    return func(a); \
 }

#define DMVNC2( NAME, A, B ) \
 mvncStatus NAME(A a, B b) { \
    typedef mvncStatus (*f)(A,B); \
    static f func=0; \
    if (!func) func = (f)LoadLibMvnc(#NAME); \
    if (!func) { return MVNC_NO_DRIVER; } \
    return func(a,b); \
 }

#define DMVNC3( NAME, A, B, C ) \
 mvncStatus NAME(A a, B b, C c) { \
    typedef mvncStatus (*f)(A,B,C); \
    static f func=0; \
    if (!func) func = (f)LoadLibMvnc(#NAME); \
    if (!func) { return MVNC_NO_DRIVER; } \
    return func(a,b,c); \
 }


#define DMVNC4( NAME, A, B, C, D ) \
 mvncStatus NAME(A a, B b, C c, D d) { \
    typedef mvncStatus (*f)(A,B,C,D); \
    static f func=0; \
    if (!func) func = (f)LoadLibMvnc(#NAME); \
    if (!func) { return MVNC_NO_DRIVER; } \
    return func(a,b,c,d ); \
 }

mvncStatus nxMvncOpenDevice(const char * a, void **b)
{
    typedef mvncStatus (*f)(const char *,void **);
    static f func=0;
    if (!func) func = (f)LoadLibMvnc("mvncOpenDevice");
    if (!func) { return MVNC_NO_DRIVER; }
    return func(a,b);
 }


DMVNC3( mvncGetDeviceName,int , char *, unsigned int );
DMVNC1( mvncCloseDevice,void *);
DMVNC4( mvncAllocateGraph,void *, void **, const void *, unsigned int );
DMVNC1( mvncDeallocateGraph,void *);
DMVNC3( mvncSetGlobalOption,int , const void *, unsigned int );
DMVNC3( mvncGetGlobalOption,int , void *, unsigned int *);
DMVNC4( mvncSetGraphOption,void *, int , const void *, unsigned int );
DMVNC4( mvncGetGraphOption,void *, int , void *, unsigned int *);
DMVNC4( mvncSetDeviceOption,void *, int , const void *, unsigned int );
DMVNC4( mvncGetDeviceOption,void *, int , void *, unsigned int *);
DMVNC4( mvncLoadTensor,void *, const void *, unsigned int , void *);
DMVNC4( mvncGetResult,void *, void **, unsigned int *, void **);

}
