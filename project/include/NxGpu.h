#ifndef NX_GPU_INCLUDED
#define NX_GPU_INCLUDED

namespace numerix
{

struct GpuData;

bool gpuInit(int inDevice=0);


GpuData *gpuAlloc(int size);

void gpuFree(GpuData *inBuffer);

void gpuUpload(GpuData *buffer, const unsigned char *inData, int n);
void gpuUploadConvert(GpuData *buffer, const unsigned char *inData, int n, bool inNchw, Tensor *ten);

void gpuDownload(unsigned char *buffer, const GpuData *inData, int n);
void gpuDownloadConvert(unsigned char *buffer, const GpuData *inData, int n, bool inNchw, Tensor *ten);


class Layer;

Layer *gpuCreateConv2D(int inStrideY, int inStrideX,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias);

Layer *gpuCreateMaxPool(int inSizeX, int inSizeY,
                       int inStrideY, int inStrideX,
                       Padding padding);

Layer *gpuCreateConcat();

Layer *gpuCreateYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh);

} // end namespace numerix


#endif

