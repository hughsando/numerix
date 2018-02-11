#include <Tensor.h>
#include <Layer.h>

#include <numerix_cuda/cuda_runtime.h>
#include <numerix_cuda/cudnn.h>

#include <NxThread.h>

#include <DynamicLoad.h>

// TODO - use runtime info
#if CUDNN_VERSION < 3000
  #define OLD_CUDNN
#else
  #define NEW_CUDNN
#endif

#ifdef HX_WINDOWS
DynamicLibrary cudaLib("cudart64_90.dll");
DynamicLibrary cudnnLib("cudnn64_7.dll");
#else
DynamicLibrary cudaLib("cudart.so");
DynamicLibrary cudnnLib("cudnn.so");
#endif


DynamicFunction0(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1, cudaGetLastError)
DynamicFunction1(cudaLib, CUDARTAPI, const char*,0, cudaGetErrorString,cudaError_t)
DynamicFunction2(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1,  cudaGetDeviceProperties,struct cudaDeviceProp *, int)
DynamicFunction1(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1,  cudaGetDevice, int *)
DynamicFunction2(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1,  cudaMalloc, void **,size_t)
DynamicFunction1(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1,  cudaFree, void *)
DynamicFunction4(cudaLib, CUDARTAPI, cudaError_t, (cudaError_t)-1,  cudaMemcpy, void *, const void *, size_t , enum cudaMemcpyKind)



DynamicFunction1(cudnnLib, CUDNNWINAPI, const char*,0, cudnnGetErrorString,cudnnStatus_t)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreate,cudnnHandle_t *)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreateTensorDescriptor,cudnnTensorDescriptor_t *)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreateFilterDescriptor,cudnnFilterDescriptor_t *)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreateConvolutionDescriptor,cudnnConvolutionDescriptor_t *)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreatePoolingDescriptor,cudnnPoolingDescriptor_t *)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnCreateActivationDescriptor,cudnnActivationDescriptor_t *)


DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnDestroyTensorDescriptor,cudnnTensorDescriptor_t )
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnDestroyFilterDescriptor,cudnnFilterDescriptor_t)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnDestroyConvolutionDescriptor,cudnnConvolutionDescriptor_t)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnDestroyPoolingDescriptor,cudnnPoolingDescriptor_t)
DynamicFunction1(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnDestroyActivationDescriptor,cudnnActivationDescriptor_t)

DynamicFunction7(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSetTensor4dDescriptor,cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int ,int, int);

DynamicFunction7(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSetFilter4dDescriptor, cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, int , int , int)

DynamicFunction9(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSetConvolution2dDescriptor, cudnnConvolutionDescriptor_t, int, int, int, int, int, int, cudnnConvolutionMode_t, cudnnDataType_t)


DynamicFunction9(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSetPooling2dDescriptor, cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, int, int, int, int, int, int)

   
DynamicFunction4(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSetActivationDescriptor, cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, double)


#ifdef OLD_CUDNN
DynamicFunction8(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnAddTensor,cudnnHandle_t,int, const void *, const cudnnTensorDescriptor_t, const void *, const void *, cudnnTensorDescriptor_t , void *)
#else

DynamicFunction7(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1,  cudnnAddTensor,cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const void *, cudnnTensorDescriptor_t, void *)
#endif

DynamicFunction8(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnGetConvolutionForwardAlgorithm, cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, size_t, cudnnConvolutionFwdAlgo_t *)


DynamicFunction7(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnGetConvolutionForwardWorkspaceSize, cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, size_t *)


DynamicFunction13(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnConvolutionForward, cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t, const void *, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *, size_t, const void *, const cudnnTensorDescriptor_t, void *)


DynamicFunction9(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnSoftmaxForward, cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void *, const cudnnTensorDescriptor_t, const void *, const void *, const cudnnTensorDescriptor_t, void *)


DynamicFunction8(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnPoolingForward, cudnnHandle_t, const cudnnPoolingDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *, const void *, const cudnnTensorDescriptor_t, void *)
   

DynamicFunction8(cudnnLib, CUDNNWINAPI, cudnnStatus_t, (cudnnStatus_t)-1, cudnnActivationForward, cudnnHandle_t, cudnnActivationDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *, const void *, const cudnnTensorDescriptor_t, void *)






namespace numerix
{

bool preferNchw = true;
cudnnTensorFormat_t preferredFormat  = CUDNN_TENSOR_NCHW;

enum Init
{
   initNone,
   initGood,
   initBad,
};


static Init isInit = initNone;

static int cudaDevice = 0;
cudnnHandle_t cudnnHandle;


void cudaCheck(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        TensorThrow(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        TensorThrow(buffer);
    }
}

void cudnnCheck(cudnnStatus_t  status)
{
   if (status!=CUDNN_STATUS_SUCCESS)
      TensorThrow( cudnnGetErrorString(status) );
}



bool gpuInit(int inDevice)
{
   if (isInit==initNone)
   {
      cudaError_t status = cudaGetDevice(&cudaDevice);
      if (status == cudaSuccess && cudaGetLastError()==cudaSuccess)
      {
         printf("got device %d\n", cudaDevice);

         cudaDeviceProp props;
         status = cudaGetDeviceProperties(&props, cudaDevice);
         cudaCheck(status);

         printf("Device : %s\n", props.name);

         printf("cudnnCreate...\n");
         cudnnCreate(&cudnnHandle);
         if (cudnnHandle)
            printf("Ok\n");
         else
            printf("No handle?\n");
         isInit = initGood;
      }
      else
      {
         printf("Could not get cuda device\n");
         isInit = initBad;
      }
   }
   return isInit==initGood;
}


GpuData *gpuAlloc(int inSize)
{
   void *result = 0;
   cudaCheck( cudaMalloc(&result, inSize) );
   return (GpuData *)result;
}

void gpuFree(GpuData *inData)
{
   if (inData)
   {
      cudaCheck( cudaFree(inData) );
   }
}


void gpuUpload(GpuData *buffer, const unsigned char *inData, int n)
{
   //printf("cpuDownload\n");
   cudaCheck( cudaMemcpy(buffer, inData, n, cudaMemcpyHostToDevice) );
}

void gpuDownload(unsigned char *buffer, const GpuData *inData, int n)
{
   //printf("gpuDownload\n");
   cudaCheck( cudaMemcpy(buffer, inData, n, cudaMemcpyDeviceToHost) );
}




static void *gpuWorkspace = 0;
size_t gpuWorkspaceSize = 0;
void *gpuGetWorkspace(size_t inSize)
{
   if (inSize>gpuWorkspaceSize)
   {
      if (gpuWorkspace)
      {
         cudaFree(gpuWorkspace);
         gpuWorkspace = 0;
      }

      gpuWorkspaceSize = inSize;

      cudaCheck( cudaMalloc(&gpuWorkspace, gpuWorkspaceSize) );
   }
   return gpuWorkspace;
}


static std::vector<u8> cpuWorkspace;

void gpuUploadConvert(GpuData *buffer, const unsigned char *inData, int n, bool inNchw, Tensor *ten)
{
   cpuWorkspace.resize(n);
   if (inNchw)
      ten->convertToNchw(&cpuWorkspace[0], inData);
   else
      ten->convertToNhwc(&cpuWorkspace[0], inData);
   cudaCheck( cudaMemcpy( buffer, &cpuWorkspace[0], n, cudaMemcpyHostToDevice) );
}

void gpuDownloadConvert(unsigned char *buffer, const GpuData *inData, int n, bool inNchw, Tensor *ten)
{
   cpuWorkspace.resize(n);
   cudaCheck( cudaMemcpy( &cpuWorkspace[0], inData, n, cudaMemcpyDeviceToHost) );
   if (inNchw)
      ten->convertToNchw(buffer, &cpuWorkspace[0]);
   else
      ten->convertToNhwc(buffer, &cpuWorkspace[0]);
}



class CudaConv2D : public Conv2DBase
{
   int padX;
   int padY;

   cudnnTensorDescriptor_t srcDesc;
   cudnnTensorDescriptor_t destDesc;
   cudnnFilterDescriptor_t weightDesc;
   cudnnTensorDescriptor_t biasDesc;
   #ifdef NEW_CUDNN
   cudnnActivationDescriptor_t activationDesc;
   #endif

   cudnnConvolutionDescriptor_t convDesc;


public:
   CudaConv2D(int inStrideY, int inStrideX, bool inIsDeconvolution,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
      : Conv2DBase(inStrideY, inStrideX, inIsDeconvolution, inActivation, inPadding,  inWeights, 0, inBias)
   {
      cudnnCheck( cudnnCreateTensorDescriptor(&srcDesc) );
      cudnnCheck( cudnnCreateTensorDescriptor(&destDesc) );
      cudnnCheck( cudnnCreateFilterDescriptor(&weightDesc) );
      cudnnCheck( cudnnCreateTensorDescriptor(&biasDesc) );

      cudnnCheck( cudnnCreateConvolutionDescriptor(&convDesc) );

      padX = inPadding==padValid ? 0 : ( (filterX-1)/2 );
      padY = inPadding==padValid ? 0 : ( (filterY-1)/2 );

      #if(CUDNN_MAJOR >= 6)
      // cudnn 6.0
      cudnnCheck( cudnnSetConvolution2dDescriptor(convDesc, padX, padY, strideX, strideY, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) );
      #else
      // cudnn 5.1
      cudnnCheck( cudnnSetConvolution2dDescriptor(convDesc,  padX, padY, strideX, strideY, 1, 1,
                                    CUDNN_CROSS_CORRELATION) );
      #endif

      #ifdef NEW_CUDNN
      cudnnCheck( cudnnCreateActivationDescriptor(&activationDesc) );
      if (activation == actSigmoid || activation==actRelu || activation==actLeaky)
      {
          cudnnCheck( cudnnSetActivationDescriptor(activationDesc,
                                activation==actSigmoid ? CUDNN_ACTIVATION_SIGMOID : CUDNN_ACTIVATION_RELU,
                                CUDNN_NOT_PROPAGATE_NAN,
                                1.0 ) );
      }
      #endif

   }

   ~CudaConv2D()
   {
      cudnnDestroyTensorDescriptor(srcDesc);
      cudnnDestroyTensorDescriptor(destDesc);
      cudnnDestroyFilterDescriptor(weightDesc);
      cudnnDestroyTensorDescriptor(biasDesc);

      cudnnDestroyConvolutionDescriptor(convDesc);

      #ifdef NEW_CUDNN
      cudnnDestroyActivationDescriptor(activationDesc);
      #endif
   }


   void doRun(Tensor *input, Tensor *output)
   {
      CShape sin = input->shape;

      int srcH = sin[0];
      int srcW = sin[1];

      int destW = (srcW + 2*padX - filterX) / strideX + 1;
      int destH = (srcH + 2*padY - filterY) / strideY + 1;


      // Load source according to what it is ...
      bool srcNchw = preferNchw; //input->isGpuNchw();
      cudnnCheck( cudnnSetTensor4dDescriptor(srcDesc, srcNchw ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, sin[2], sin[0], sin[1]) );

      // Dest according to preference...
      cudnnCheck( cudnnSetTensor4dDescriptor(destDesc, preferredFormat, CUDNN_DATA_FLOAT, 1, outputs, destH, destW) );

      // Weight according to preference...
      #ifdef NEW_CUDNN
      cudnnCheck( cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, preferredFormat, outputs, inputs, filterY, filterX) );
      #else
      cudnnCheck( cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, outputs, inputs, filterY, filterX) );
      #endif

      cudnnConvolutionFwdAlgo_t algo;
      cudnnCheck( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
            srcDesc,
            weightDesc,
            convDesc,
            destDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &algo) );
      size_t workSize = 0;
      cudnnCheck( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcDesc,
                weightDesc,
                convDesc,
                destDesc,
                algo,
                &workSize) );

      void *workspace = gpuGetWorkspace(workSize);

      float alpha = 1;
      float beta  = 0;

      cudnnConvolutionForward(cudnnHandle,
                &alpha,
                srcDesc,
                input->gpuRead(srcNchw),
                weightDesc,
                weights->gpuRead(preferNchw),
                convDesc,
                algo,
                workspace,
                workSize,
                &beta,
                destDesc,
                output->gpuWrite(preferNchw) );

      // TODO - cudnnConvolutionBiasActivationForward, but what is z for?

      if (bias)
      {
         // Bias according to preference - no need to re-order
         cudnnCheck( cudnnSetTensor4dDescriptor(biasDesc, preferredFormat, CUDNN_DATA_FLOAT, 1, outputs, 1, 1) );



         beta = 1;
         cudnnAddTensor( cudnnHandle,
                        #ifdef OLD_CUDNN
                        CUDNN_ADD_SAME_C,
                        #endif
                        &alpha,
                        biasDesc,
                        bias->gpuRead(),
                        &beta,
                        destDesc,
                        output->gpuWrite(preferNchw) );
      }

      if (activation==actRelu || activation==actSigmoid || activation==actLeaky)
      {
         float alpha = 1.0;
         float beta =  0.0;
         if (activation==actLeaky)
         {
            alpha = 0.9;
            beta = 0.1;
         }
         cudnnCheck( cudnnActivationForward( cudnnHandle,
                       #ifdef OLD_CUDNN
                       activation==actSigmoid ? CUDNN_ACTIVATION_SIGMOID : CUDNN_ACTIVATION_RELU,
                       #else
                       activationDesc,
                       #endif
                       &alpha,
                       destDesc,
                       output->gpuRead(preferNchw),
                       &beta,
                       destDesc,
                       output->gpuWrite(preferNchw) ) );
      }
   }
};

Layer *gpuCreateConv2D(int inStrideY, int inStrideX, bool inIsDeconvolution,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias)
{
   return new CudaConv2D(inStrideY, inStrideX, inIsDeconvolution, activation, padding, weights, bias);
}









class CudaMaxPool : public Layer
{
   int        filterY;
   int        filterX;
   int        strideX;
   int        strideY;

   Padding    padding;
   int        padX;
   int        padY;

   cudnnTensorDescriptor_t srcDesc;
   cudnnTensorDescriptor_t destDesc;
   cudnnPoolingDescriptor_t poolingDesc;

public:
   CudaMaxPool(int inSizeX, int inSizeY,
           int inStepX, int inStepY,
           Padding inPadding )
   {
      filterX = inSizeX;
      filterY = inSizeY;
      strideX = inStepX;
      strideY = inStepY;
      padding = inPadding;

      padX = padding==padValid ? 0 : ( (filterX-1)/2 );
      padY = padding==padValid ? 0 : ( (filterY-1)/2 );


      cudnnCheck( cudnnCreateTensorDescriptor(&srcDesc) );
      cudnnCheck( cudnnCreateTensorDescriptor(&destDesc) );

      cudnnCreatePoolingDescriptor(&poolingDesc);

      cudnnCheck( cudnnSetPooling2dDescriptor( poolingDesc,
                 CUDNN_POOLING_MAX,
                 #ifdef NEW_CUDNN
                 CUDNN_NOT_PROPAGATE_NAN,
                 #endif
                 filterY,
                 filterX,
                 padY,
                 padX,
                 strideY,
                 strideX) );
   }

   ~CudaMaxPool()
   {
      cudnnDestroyTensorDescriptor(srcDesc);
      cudnnDestroyTensorDescriptor(destDesc);
      cudnnDestroyPoolingDescriptor(poolingDesc);
   }


   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("CudaMaxPool only supports Float32 tensors");

      CShape &sin = inSrc0->shape;
      if (sin.size()!=3)
         TensorThrow("CudaMaxPool only supports H*W*C tensors");

      int srcH = sin[0];
      int srcW = sin[1];
      int channels = sin[2];

      int destW = (srcW + 2*padX - filterX) / strideX + 1;
      int destH = (srcH + 2*padY - filterY) / strideY + 1;

      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape &s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==channels;
      }
      Tensor *result = inBuffer;
      if (!match)
         result = new Tensor( Float32, Shape3(destH, destW, channels ) );

      cudnnCheck( cudnnSetTensor4dDescriptor(srcDesc, preferredFormat, CUDNN_DATA_FLOAT, 1, channels, srcH, srcW) );
      cudnnCheck( cudnnSetTensor4dDescriptor(destDesc, preferredFormat, CUDNN_DATA_FLOAT, 1, channels, destH, destW) );


      float alpha = 1;
      float beta = 0;
      cudnnCheck( cudnnPoolingForward( cudnnHandle,
                    poolingDesc,
                    &alpha,
                    srcDesc,
                    inSrc0->gpuRead(preferNchw),
                    &beta,
                    destDesc,
                    result->gpuWrite(preferNchw) ) );

      return result;
   }
};

Layer *gpuCreateMaxPool(int sizeX, int sizeY,
                        int stepX, int stepY,
                        Padding padding )
{
   return new CudaMaxPool(sizeX, sizeY, stepX, stepY, padding);
}





class CudaConcat : public Layer
{
public:
   CudaConcat( ) { }


   Tensor *run(Tensor *inSrc0, Tensor *inSrc1, Tensor *inBuffer)
   {
      if (inSrc0->type != inSrc1->type)
         TensorThrow("Concat - input types must match");

      CShape sin0 = inSrc0->shape;
      CShape sin1 = inSrc1->shape;
      if (sin0.size()!=3 || sin1.size()!=3)
         TensorThrow("Concat only supports H*W*C tensors");

      if (sin0[0]!=sin1[0] || sin0[0]!=sin1[0])
         TensorThrow("Concat - mismatch image sizes");

      int srcH = sin0[0];
      int srcW = sin0[1];
      int c0 = sin0[2];
      int c1 = sin1[2];
      int channels = c0 + c1;
      //printf("################### Concat -> %d %d %d\n", srcW, srcH, channels);

      Tensor *result = Tensor::makeBuffer(inBuffer, srcW, srcH, channels, inSrc0->type);

      const u8 *s0 = inSrc0->gpuRead(true);
      const u8 *s1 = inSrc1->gpuRead(true);

      u8 *d = result->gpuWrite(true);


      cudaCheck( cudaMemcpy(d, s0, inSrc0->getByteCount(), cudaMemcpyDeviceToDevice ) );

      d += inSrc0->getByteCount();
      cudaCheck( cudaMemcpy(d, s1, inSrc1->getByteCount(), cudaMemcpyDeviceToDevice ) );

      return result;
   }
};

Layer *gpuCreateConcat()
{
   return new CudaConcat();
}




class CudaYoloLayer : public Layer
{
   std::vector<float> anchors;
   int boxCount;
   int classCount;
   float thresh;
   Boxes boxes;
   std::vector<Boxes> boxbuffers;
   cudnnTensorDescriptor_t softDesc;


   inline float logistic_activate(float f) { return 1.0 / (1.0 + exp(-f)); }


public:
   CudaYoloLayer(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh) :
      anchors(inAnchors), boxCount(inBoxCount), classCount(inClassCount)
   {
      thresh = inThresh;

      cudnnCheck( cudnnCreateTensorDescriptor(&softDesc) );
   }
   ~CudaYoloLayer()
   {
      cudnnDestroyTensorDescriptor(softDesc);
   }

   Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      boxes.resize(0);
      int h = inSrc0->shape[0];
      int w = inSrc0->shape[1];
      int channels = inSrc0->shape[2];
      std::vector<float> softmaxBuf(classCount);

      Tensor *softMaxxed = Tensor::makeBuffer(inBuffer, w, h, boxCount*(5+classCount), Float32);


      // Gpu format will be channel-major arrays of:
      //
      //  [scale, bx, by, bw, bh,  c0, c1, ..... cN-1] * boxCount
      const u8 *input = inSrc0->gpuRead(true);
      u8 *dest  = softMaxxed->gpuWrite(true);

      // We want to copy the box channels, and softmax the class channels
      int boxBytes = 5*w*h*sizeof(float);
      int classBytes = w*h*classCount*sizeof(float);

      cudnnCheck( cudnnSetTensor4dDescriptor(softDesc, preferredFormat, CUDNN_DATA_FLOAT, 1, classCount, w, h) );

      for(int b=0;b<boxCount;b++)
      {
         cudaMemcpy( dest, input, boxBytes, cudaMemcpyDeviceToDevice);
         dest += boxBytes;
         input += boxBytes;

         float one = 1;
         float zero = 0;
         cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &one, softDesc, input, &zero, softDesc, dest);

         dest += classBytes;
         input += classBytes;
      }

      // Read channel-minor format ..
      src = (const float *)softMaxxed->cpuRead(false);
      srcW = w;
      srcH = h;
      boxbuffers.resize( GetWorkerCount() );
      runThreaded();
      src = 0;

      boxes = boxbuffers[0];
      for(int i=1;i<boxbuffers.size();i++)
      {
         Boxes &b = boxbuffers[i];
         for(int j=0;j<b.size();j++)
            boxes.push_back(b[j]);
      }
      SortBoxes(boxes);

      return softMaxxed;

   }

   const float *src;
   int srcW;
   int srcH;

   void runThread(int threadId)
   {
      Boxes &boxen = boxbuffers[threadId];
      boxen.resize(0);

      int boxSize = 4 + 1 + classCount;

      while(true)
      {
         int y = getNextJob();
         if (y>=srcH)
            break;

         for(int x=0;x<srcW;x++)
         {
            const float *predictions = src + (y*srcW + x) * boxSize*boxCount;

            for(int b=0;b<boxCount;b++)
            {
               const float *box = predictions;
               float scale = logistic_activate( predictions[4] );
               const float *classes = predictions + 5;

               int foundClass = -1;
               float foundBest = 0;
               for(int j=0;j<classCount;j++)
               {
                  if (classes[j]>foundBest)
                  {
                     foundClass = j;
                     foundBest = classes[j];
                  }
               }
               float prob = scale * foundBest;

               if (prob>thresh)
               {
                  BBox bbox;
                  bbox.classId = foundClass;
                  bbox.prob = prob;
                  bbox.x = (x + logistic_activate(box[0]))/srcW;
                  bbox.y = (y + logistic_activate(box[1]))/srcH;
                  bbox.w = exp(box[2]) * anchors[2*b] / srcW;
                  bbox.h = exp(box[3]) * anchors[2*b+1] / srcH;

                  boxen.push_back(bbox);
               }

               predictions += boxSize;
            }
         }
      }
   }

   void getBoxes(Boxes &outBoxes) { outBoxes = boxes; }
};

Layer *gpuCreateYolo(const std::vector<float> &inAnchors,int inBoxCount, int inClassCount, float inThresh)
{
   return new CudaYoloLayer(inAnchors, inBoxCount, inClassCount, inThresh);
}







} // end namespace numerix







