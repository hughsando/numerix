#ifndef OCL_INCLUDED
#define OCL_INCLUDED

#include <string>
#include <vector>

struct _cl_mem;

namespace numerix
{

struct OclProp
{
   std::string key;
   std::string value;
};
typedef std::vector<OclProp> OclProps;
typedef std::vector<void *> OclDeviceList;
typedef std::vector<void *> OclPlatformList;

OclPlatformList oclGetPlatformList();
void oclGetPlatformProps(void *inPlatform, OclProps &outProps);
OclDeviceList oclGetPlatformDevices(void *inPlatform);
void oclGetDeviceProps(void *inDevice, OclProps &outProps, int &outComputeUnits);


class Layer;

Layer *oclCreateConv2D(int inStrideY, int inStrideX,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias);

Layer *oclCreateMaxPool(int inSizeX, int inSizeY,
                       int inStrideY, int inStrideX,
                       Padding padding);

Layer *oclCreateConcat();

Layer *oclCreateYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh);



typedef _cl_mem OclData;


OclData *oclAlloc(int size);
void oclFree(OclData *inBuffer);

void oclDownload(unsigned char *buffer, const OclData *inData, int n);
void oclUpload(OclData *buffer, const unsigned char *inData, int n);


class OclContext
{
   public:
      static OclContext *create(void *inPlatform, OclDeviceList &inDevices);

      static void setCurrent(OclContext *inContext);
      static bool hasCurrent();

      virtual void incRef() = 0;
      virtual void decRef() = 0;

   protected:
      virtual ~OclContext() { }
};

}

#endif

