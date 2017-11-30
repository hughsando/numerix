#include <Tensor.h>
#include <Layer.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#if CUDNN_VERSION < 3000
  #define OLD_CUDNN
#else
  #define NEW_CUDNN
#endif

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



class CudaConv2D : public Layer
{
   int        strideX;
   int        strideY;
   int        filterY;
   int        filterX;
   int        padX;
   int        padY;
   int        inputs;
   int        outputs;

   Activation activation;
   Padding    padding;
   Tensor     *weights;
   Tensor     *bias;

   cudnnTensorDescriptor_t srcDesc;
   cudnnTensorDescriptor_t destDesc;
   cudnnFilterDescriptor_t weightDesc;
   cudnnTensorDescriptor_t biasDesc;
   #ifdef NEW_CUDNN
   cudnnActivationDescriptor_t activationDesc;
   #endif

   cudnnConvolutionDescriptor_t convDesc;


public:
   CudaConv2D(int inStrideY, int inStrideX,
          Activation inActivation, Padding inPadding,
          Tensor *inWeights, Tensor *inBias)
   {
      strideY = inStrideY;
      strideX = inStrideX;
      activation = inActivation;
      padding = inPadding;

      // Output x Height x Width x Input
      CShape s = inWeights->shape;
      if (s.size()!=4)
         TensorThrow("Invalid Conv2D weight shape");

      outputs = s[0];
      filterY = s[1];
      filterX = s[2];
      inputs =  s[3];

      if (inBias && inBias->shape.size()!=1)
         TensorThrow("Conv2D - bias should be one dimensional");

      if (inBias && inBias->shape[0]!=outputs)
         TensorThrow("Conv2D - bias does not match output size");


      weights = inWeights->incRef();
      bias = inBias ? inBias->incRef() : 0;

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


      weights->decRef();
      if (bias)
         bias->decRef();
   }

   void setNormalization(Tensor *inScales, Tensor *inMeans, Tensor *inVars)
   {
      // a set of output features are first "normalized":
      //
      // O'i = (Oi - Mean_i)/(sqrt(Vars_i) +  .000001f)
      //
      // Then 'scale_bias'
      //
      // O''i = O'i * Scale_i
      //
      // Let Ki = Scale_i/(sqrt(Vars_i) +  .000001f)
      //     Ci = -Mean_i * Ki
      //
      // O' = Ki Oi + Ci
      // This is the same as multiplying the Weights_i by Ki and adding Ci to the biases

      const float *scale = (const float *)inScales->cpuRead();
      const float *mean = (const float *)inMeans->cpuRead();
      const float *var = (const float *)inVars->cpuRead();

      if (!bias)
      {
         Shape s(1);
         s[0]=outputs;
         bias = new Tensor(Float32, s);
         bias->zero(0,outputs);
      }
      float *b = (float *)bias->cpuWritePart();
      float *w = (float *)weights->cpuWritePart();

      int wCount = weights->strides[0];
      for(int i=0;i<outputs;i++)
      {
         float Ki = scale[i]/(sqrt(var[i])+.000001f);
         for(int i=0;i<wCount;i++)
            *w++ *= Ki;
         b[i] -= mean[i]*Ki;
      }

   }

   virtual Tensor *run(Tensor *inSrc0, Tensor *inBuffer)
   {
      if (inSrc0->type != Float32)
         TensorThrow("Conv2D only supports Float32 tensors");

      CShape &sin = inSrc0->shape;
      if (sin.size()!=3)
         TensorThrow("Conv2D only supports H*W*C tensors");

      if (sin[2]!=inputs)
      {
         printf("sin : %d %d %d\n", sin[0], sin[1], sin[2]);
         printf("weights : %d %dx%d %d\n", outputs, filterY, filterX, inputs );
         TensorThrow("Conv2D - weights do not match the number of input channels");
      }

      int srcH = sin[0];
      int srcW = sin[1];

      int destW = (srcW + 2*padX - filterX) / strideX + 1;
      int destH = (srcH + 2*padY - filterY) / strideY + 1;

      bool match = false;
      if (inBuffer && inBuffer->shape.size()==3 && inBuffer->type==Float32)
      {
         CShape &s = inBuffer->shape;
         match = s[0]==destH && s[1]==destW && s[2]==outputs;
      }
      Tensor *result = inBuffer;
      if (!match)
         result = new Tensor( Float32, Shape3(destH, destW, outputs ) );



      // Load source according to what it is ...
      bool srcNchw = preferNchw; //inSrc0->isGpuNchw();
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
                inSrc0->gpuRead(srcNchw),
                weightDesc,
                weights->gpuRead(preferNchw),
                convDesc,
                algo,
                workspace,
                workSize,
                &beta,
                destDesc,
                result->gpuWrite(preferNchw) );

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
                        result->gpuWrite(preferNchw) );
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
                       result->gpuRead(preferNchw),
                       &beta,
                       destDesc,
                       result->gpuWrite(preferNchw) ) );
      }

      return result;
   }
};

Layer *gpuCreateConv2D(int inStrideY, int inStrideX,
                       Activation activation, Padding padding,
                       Tensor *weights, Tensor *bias)
{
   return new CudaConv2D(inStrideY, inStrideX, activation, padding, weights, bias);
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










} // end namespace numerix







