#ifndef OCL_INCLUDED
#define OCL_INCLUDED

#include <string>
#include <vector>
#include <Tensor.h>

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

Layer *oclCreateMaxPool(int inSizeX, int inSizeY,
                        int inStrideY, int inStrideX,
                        Padding padding);

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

