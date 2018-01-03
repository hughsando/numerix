#include <DynamicLoad.h>
#include <stdio.h>

#ifdef HX_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif

DynamicLibrary::DynamicLibrary(const char *inName)
{
   #ifdef HX_WINDOWS
   handle = LoadLibraryA(inName);
   #else
   handle = dlopen(inName, RTLD_GLOBAL|RTLD_NOW);
   #endif
   //printf("Loaded library %s -> %p\n", inName, handle);

}


void *DynamicLibrary::load(const char *inName)
{
   if (!handle)
      return 0;

   #ifdef HX_WINDOWS
   void *result = GetProcAddress((HMODULE)handle, inName);
   #else
   void *result = dlsym(handle, inName);
   #endif
   //printf("%s -> %p\n", inName, result);
   return result;
}


