#ifndef NX_GPU_INCLUDED
#define NX_GPU_INCLUDED

namespace numerix
{

struct GpuData;

bool gpuInit(int inDevice=0);


GpuData *gpuAlloc(int size);

void gpuFree(GpuData *inBuffer);

void gpuUpload(GpuData *buffer, const unsigned char *inData, int n);

void gpuDownload(unsigned char *buffer, const GpuData *inData, int n);


class Layer;
class Tensor;

Layer *gpuCreateConv2D(int inStrideY, int inStrideX,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias);

Layer *gpuCreateMaxPool(int inSizeX, int inSizeY,
                       int inStrideY, int inStrideX,
                       Padding padding);

} // end namespace numerix


#endif

