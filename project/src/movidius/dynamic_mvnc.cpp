#include <stdio.h>
#include "mvnc.h"
#include <DynamicLoad.h>


extern "C" {

DynamicLibrary mvnc("libmvnc.so");


DynamicFunction2Name(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncOpenDevice, nxMvncOpenDevice, const char *, void **);
DynamicFunction3(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncGetDeviceName,int , char *, unsigned int );
DynamicFunction1(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncCloseDevice,void *);
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncAllocateGraph,void *, void **, const void *, unsigned int );
DynamicFunction1(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncDeallocateGraph,void *);
DynamicFunction3(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncSetGlobalOption,int , const void *, unsigned int );
DynamicFunction3(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncGetGlobalOption,int , void *, unsigned int *);
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncSetGraphOption,void *, int , const void *, unsigned int );
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncGetGraphOption,void *, int , void *, unsigned int *);
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncSetDeviceOption,void *, int , const void *, unsigned int );
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncGetDeviceOption,void *, int , void *, unsigned int *);
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncLoadTensor,void *, const void *, unsigned int , void *);
DynamicFunction4(mvnc,,mvncStatus,MVNC_NO_DRIVER, mvncGetResult,void *, void **, unsigned int *, void **);

}
