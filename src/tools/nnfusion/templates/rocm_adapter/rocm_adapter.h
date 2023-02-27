// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __HIP_RUNTIME_H__
#define __HIP_RUNTIME_H__

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#undef __HIP_PLATFORM_NVCC__

#include <hip/hip_runtime.h>
#define ROCM_USE_FLOAT16
#include <rocblas/rocblas.h>
// #include <hipsparse.h>
// #include <hiprand/hiprand.h>
// #include <hiprand_kernel.h>
#include <miopen/miopen.h>
#include <miopen/version.h>
#include <hip/hip_complex.h>

#include <unordered_map>
#include <vector>

#include <hip/hcc_detail/hip_fp16_math_fwd.h>
#define half _Float16
#define htanh tanhf
#define htan tanf
#define hatan atanf
#define herf erff
#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16
#define hexp __ocml_exp_f16
#define __float2half_rn(x) _Float16(x)

#define __ll2half_rn(x) _Float16(x)
#define __half2ll_rn(x) int64_t(x)

#define ensure_cmd(x)  ((x) || (printf("ensure_cmdion Error %s(L-%d): %s\n", __FILE__, __LINE__, #x), abort(), false))

#undef assert
#define assert ensure_cmd

#define UN_IMPLEMENTED(func, ret_type)   inline ret_type func(...) { printf("Function `%s` not implemented for rocm.\n", __func__); ensure_cmd(0); return (ret_type)0; }

#ifndef __PARSING_KERNEL_ARGS__
#define __PARSING_KERNEL_ARGS__
#define ARG_2(b, t)         b, t, 0, 0
#define ARG_3(b, t, m)      b, t, m, 0
#define ARG_4(b, t, m, s)   b, t, m, s
#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define KERNEL_CONFIG(...) GET_MACRO(__VA_ARGS__, ARG_4, ARG_3, ARG_2)(__VA_ARGS__)
#endif

template<class T> static inline T& getVariable(const void *key);

/*********************************************************
          CUDA Driver Translation
*********************************************************/

#define CUDA_ERROR_DEINITIALIZED hipErrorDeinitialized
#define cuMemsetD32 hipMemsetD32
#define cuModuleUnload hipModuleUnload
#define cuModuleLoadData hipModuleLoadData
#define cuModuleGetFunction hipModuleGetFunction
#define cuGetErrorName(x, y) (*(y) = hipGetErrorName(x), hipSuccess)

#define CUmodule hipModule_t
#define CU_STREAM_LEGACY 0
#define CUstream hipStream_t
#define CUevent hipEvent_t
#define CUcontext size_t
#define CUdeviceptr hipDeviceptr_t
#define cuInit hipInit
#define cuDeviceGetCount hipGetDeviceCount
#define cuDevicePrimaryCtxRetain(pctx, dev)  (*(pctx) = (dev), hipSuccess)
#define cuCtxSetCurrent hipSetDevice
#define cuCtxGetCurrent hipGetDevice
#define cuStreamCreate hipStreamCreateWithFlags
#define CU_STREAM_NON_BLOCKING 0 // hipStreamNonBlocking
#define cuStreamSynchronize hipStreamSynchronize
#define cuMemAlloc_v2 hipMalloc
#define cuMemFree_v2 hipFree
#define cuMemcpyHtoDAsync_v2(dst, src, bytes, st) hipMemcpyHtoDAsync(dst, (void*)(src), bytes, st)
#define cuMemcpyDtoHAsync_v2 hipMemcpyDtoHAsync
#define cuMemcpyDtoDAsync_v2 hipMemcpyDtoDAsync
#define CUresult hipError_t
#define cuEventQuery hipEventQuery
#define cuEventDestroy hipEventDestroy
#define CUfunction hipFunction_t
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventRecord hipEventRecord

inline CUresult cuEventCreate(CUevent *event, int flags = 0) {
  return hipEventCreateWithFlags(event, flags);
}

#define CUDA_SUCCESS hipSuccess
#define CUDA_ERROR_NOT_READY hipErrorNotReady

#define cuMemsetD32Async(dst, ui, count, stream) hipMemsetD32Async(dst, ui, count, stream) // __cuMemsetD32Async(dst, ui, count, stream)
    /* ({
      cudnnTensorDescriptor_t yDesc;
      ensure_cmd(0 == cudnnCreateTensorDescriptor(&yDesc));
      ensure_cmd(0 == cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, miopenInt32, 1, count, 1, 1));
      int alpha = (ui);
      ensure_cmd(0 == miopenSetTensor(NULL, yDesc, dst, &alpha));
      ensure_cmd(0 == cudnnDestroyTensorDescriptor(yDesc));
    }) */
#define cuModuleGetGlobal(...) (assert(0), hipSuccess)
#define cuLaunchKernel hipLaunchKernel

/*********************************************************
          CUDA Runtime Translation
*********************************************************/

#define cudaGetErrorName hipGetErrorName
#define cudaErrorCudartUnloading hipErrorNotInitialized
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxBlockDimX hipDeviceAttributeMaxBlockDimX
#define cudaDevAttrMaxBlockDimY hipDeviceAttributeMaxBlockDimY
#define cudaDevAttrMaxBlockDimZ hipDeviceAttributeMaxBlockDimZ

#define cudaStreamSynchronize cuStreamSynchronize
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaStreamCreate(x) cuStreamCreate(x, 0)
#define cudaMemcpyKind hipMemcpyKind
#define sharedMemPerMultiprocessor maxSharedMemoryPerMultiProcessor
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaGetLastError hipGetLastError
#define cudaStreamDefault 0
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaError hipError_t
#define cudaSuccess hipSuccess
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaMemcpy hipMemcpy
#define cudaEventRecord hipEventRecord
#define cudaEventCreate cuEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaEventSynchronize hipEventSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventDestroy hipEventDestroy
#define cudaEventQuery hipEventQuery
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetErrorString hipGetErrorString
#define cudaStreamWaitEvent hipStreamWaitEvent

#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties

/*********************************************************
          CUBLAS Translation
*********************************************************/

#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasSetStream rocblas_set_stream
#define cublasGetStream rocblas_get_stream
#define cublasHandle_t rocblas_handle
#define cublasStatus_t rocblas_status
#define cublasSetPointerMode(hd, mode) (ensure_cmd((mode) == rocblas_pointer_mode_host), rocblas_status_success) // rocblas_set_pointer_mode
#define cublasPointerMode_t rocblas_pointer_mode
#define CUBLAS_POINTER_MODE_HOST rocblas_pointer_mode_host
#define CUBLAS_POINTER_MODE_DEVICE rocblas_pointer_mode_device

#define cublasSgemm rocblas_sgemm
#define cublasHgemm rocblas_hgemm
#define cublasSaxpy rocblas_saxpy
#define cublasSscal rocblas_sscal
#define cublasSgemmStridedBatched rocblas_sgemm_strided_batched

#define CUDA_R_64F rocblas_datatype_f64_r // HIPBLAS_R_64F
#define CUDA_R_32F rocblas_datatype_f32_r // HIPBLAS_R_32F
#define CUDA_R_16F rocblas_datatype_f16_r // HIPBLAS_R_16F

#define CUBLAS_STATUS_SUCCESS           rocblas_status_success // HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED   rocblas_status_invalid_handle // HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED     rocblas_status_not_implemented // HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_INVALID_VALUE     rocblas_status_invalid_pointer // HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_MAPPING_ERROR     rocblas_status_invalid_size // HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_ALLOC_FAILED      rocblas_status_memory_error // HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR    rocblas_status_internal_error // HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED  (cublasStatus_t)(-3) // HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH     (cublasStatus_t)(-2) // HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_LICENSE_ERROR     (cublasStatus_t)(-1)


#define CUBLAS_OP_N rocblas_operation_none // HIPBLAS_OP_N
#define CUBLAS_OP_T rocblas_operation_transpose // HIPBLAS_OP_T

#define cublasGemmAlgo_t                 rocblas_gemm_algo // hipblasGemmAlgo_t
#define CUBLAS_GEMM_DFALT                rocblas_gemm_algo_standard // HIPBLAS_GEMM_DEFAULT
#define cudaDataType                     rocblas_datatype // hipblasDatatype_t
#define cublasOperation_t                rocblas_operation_ // hipblasOperation_t

#define cublasCreate_v2 cublasCreate
#define cublasDestroy_v2 cublasDestroy
#define cublasSetStream_v2 cublasSetStream
#define cublasSetPointerMode_v2 cublasSetPointerMode
#define cublasSgemm_v2 hipblasSgemm
#define cublasSetMathMode(...) CUBLAS_STATUS_SUCCESS
#define CUBLAS_TENSOR_OP_MATH 0


#define NULL_IMPL(...) (assert(0), CUBLAS_STATUS_SUCCESS)
#define USING(x) NULL_IMPL

#define cublasSgemmBatched USING(hipblasSgemmBatched)
#define cublasDgemmBatched USING(hipblasDgemmBatched)
#define cublasHgemmBatched USING(hipblasHgemmBatched)
#define cublasSgeam USING(hipblasSgeam)
#define cublasDgeam USING(hipblasDgeam)
#define cublasHgeam USING(hipblasHgeam)
#define cublasScopy USING(hipblasScopy)
#define cublasDcopy USING(hipblasDcopy)
#define cublasHcopy USING(hipblasHcopy)
#define cublasSdot USING(hipblasSdot)
#define cublasDdot USING(hipblasDdot)
#define cublasHdot USING(hipblasHdot)
#define cublasDscal USING(hipblasDscal)
#define cublasHscal USING(hipblasHscal)
#define cublasDaxpy USING(hipblasDaxpy)
#define cublasHaxpy USING(hipblasHaxpy)
#define cublasDgemm USING(hipblasDgemm)
#define cublasSasum USING(hipblasSasum)
#define cublasDasum USING(hipblasDasum)
#define cublasHasum USING(hipblasHasum)
#define cublasIsamax USING(hipblasIsamax)
#define cublasIdamax USING(hipblasIdamax)

#define cublasScalEx USING(hipblasScalEx)
#define cublasGemmEx USING(hipblasGemmEx)
#define cublasAxpyEx USING(hipblasAxpyEx)
#define cublasDotEx USING(hipblasDotEx)

UN_IMPLEMENTED(hipblasSgemmBatched, cublasStatus_t)
UN_IMPLEMENTED(hipblasDgemmBatched, cublasStatus_t)
UN_IMPLEMENTED(hipblasHgemmBatched, cublasStatus_t)

UN_IMPLEMENTED(hipblasScalEx, cublasStatus_t)
UN_IMPLEMENTED(hipblasAxpyEx, cublasStatus_t)
UN_IMPLEMENTED(hipblasDotEx, cublasStatus_t)


/*********************************************************
          CURAND Translation
*********************************************************/

#define curand_uniform hiprand_uniform
#define curand_normal hiprand_normal
#define curand_init hiprand_init
#define curandState hiprandState_t
#define curandStatus_t hiprandStatus_t
#define curandStatus hipsparseStatus_t
#define curandGenerator_t hiprandGenerator_t
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble

/*********************************************************
          CUDNN Translation
*********************************************************/

typedef enum {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2
} cudnnTensorFormat_t;

typedef enum {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0, /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1  /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

#define CUDNN_ACTIVATION_RELU miopenActivationRELU
#define cudnnActivationForward miopenActivationForward
#define cudnnActivationBackward miopenActivationBackward
#define cudnnActivationMode_t miopenActivationMode_t
#define cudnnSetActivationDescriptor(d, mode, nanopt, a) miopenSetActivationDescriptor(d, mode, a, 0, 0)
#define cudnnOpTensorDescriptor_t miopenTensorOp_t
#define cudnnCreateOpTensorDescriptor(x)  miopenStatusSuccess
#define cudnnDestroyOpTensorDescriptor(x)  miopenStatusSuccess
#define cudnnSetOpTensorDescriptor(desc, op_t, dtype, nanopt) \
        ((desc) = (op_t), (dtype) == miopenFloat ? miopenStatusSuccess : miopenStatusNotImplemented)
#define CUDNN_OP_TENSOR_ADD miopenTensorOpAdd
#define CUDNN_OP_TENSOR_MUL miopenTensorOpMul
#define CUDNN_OP_TENSOR_MIN miopenTensorOpMin
#define CUDNN_OP_TENSOR_MAX miopenTensorOpMax
#define CUDNN_OP_TENSOR_SQRT ((cudnnOpTensorOp_t)-1)
#define CUDNN_OP_TENSOR_NOT ((cudnnOpTensorOp_t)-2)
#define cudnnOpTensor miopenOpTensor

#define cudnnActivationDescriptor_t miopenActivationDescriptor_t
#define cudnnCreateActivationDescriptor miopenCreateActivationDescriptor
#define cudnnDestroyActivationDescriptor miopenDestroyActivationDescriptor

#define CUDNNWINAPI
#define cudnnDataType_t miopenDataType_t
#define cudnnHandle_t miopenHandle_t
#define cudnnStatus_t miopenStatus_t

#define CUDNN_BATCHNORM_SPATIAL miopenBNSpatial
#define CUDNN_STATUS_SUCCESS miopenStatusSuccess
#define cudnnDeriveBNTensorDescriptor miopenDeriveBNTensorDescriptor
#define cudnnGetErrorString miopenGetErrorString

#define CUDNN_PROPAGATE_NAN 0
#define cudnnNanPropagation_t int
#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_STATUS_NOT_SUPPORTED miopenStatusNotImplemented
#define cudnnTensorDescriptor_t miopenTensorDescriptor_t

inline cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  return miopenCreateTensorDescriptor(tensorDesc);
}

inline cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  return miopenDestroyTensorDescriptor(tensorDesc);
}

inline cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
  if (format != CUDNN_TENSOR_NCHW)
	return miopenStatusNotImplemented;
  return miopenSet4dTensorDescriptor(tensorDesc, dataType, n, c, h, w);
}

inline cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType, int *n, int *c, int *h, int *w, int *sn, int *sc, int *sh, int *sw) {
  return miopenGet4dTensorDescriptor(tensorDesc, dataType, n, c, h, w, sn, sc, sh, sw);
}


#define cudnnBatchNormalizationForwardTraining(hd, mod, a, b, x, d_x, y, d_y, bn, d_s, d_b, factor, d_rm, r_rv, eps, d_sm, d_sv) \
          miopenBatchNormalizationForwardTraining(hd, mod, (void*)a, (void*)b, x, (void*)d_x, y, d_y, bn, (void*)d_s, (void*)d_b, factor, d_rm, r_rv, eps, d_sm, d_sv)
#define cudnnBatchNormalizationForwardInference(hd, mode, a, b, x, d_x, y, d_y, bn, d_s, d_b, d_em, d_ev, eps) \
          miopenBatchNormalizationForwardInference(hd, mode, (void*)a, (void*)b, x, (void*)d_x, y, (void*)d_y, bn, (void*)d_s, (void*)d_b, (void*)d_em, (void*)d_ev, eps)
#define cudnnBatchNormalizationBackward(hd, mode, ad, bd, ap, bp, d, d_x, dy, d_dy, dx, d_dx, bn, b_s, d_rs, d_rv, eps, d_sm, d_sv) \
          miopenBatchNormalizationBackward(hd, mode, ad, bd, ap, bp, d, d_x, dy, d_dy, dx, d_dx, bn, b_s, d_rs, d_rv, eps, d_sm, d_sv)
#define cudnnFilterDescriptor_t miopenTensorDescriptor_t
#define cudnnPoolingDescriptor_t miopenPoolingDescriptor_t

#define cudnnCreateFilterDescriptor cudnnCreateTensorDescriptor
#define cudnnDestroyFilterDescriptor cudnnDestroyTensorDescriptor

#define cudnnCreatePoolingDescriptor miopenCreatePoolingDescriptor
#define cudnnDestroyPoolingDescriptor miopenDestroyPoolingDescriptor

#define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
#define cudnnConvolutionMode_t miopenConvolutionMode_t
#define CUDNN_CROSS_CORRELATION miopenConvolution
#define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM miopenConvolutionFwdAlgoGEMM
#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define cudnnDestroyConvolutionDescriptor miopenDestroyConvolutionDescriptor
#define cudnnSetConvolution2dDescriptor(d, p0, p1, s0, s1, d0, d1, mode, type) \
	miopenInitConvolutionDescriptor(d, mode, p0, p1, s0, s1, d0, d1)
#define cudnnSetConvolutionMathType(...) miopenStatusSuccess

#define cudnnPoolingMode_t miopenPoolingMode_t
#define CUDNN_POOLING_MAX miopenPoolingMax
#define CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING miopenPoolingAverage
#define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING miopenPoolingAverageInclusive
#define cudnnGetPooling2dDescriptor(d, m, nanopt, w0, w1, p0, p1, s0, s1) \
    (*(nanopt) = 0, miopenGet2dPoolingDescriptor(d, m, w0, w1, p0, p1, s0, s1))

#define cudnnSetPooling2dDescriptor(desc, mode, nanOpt, k0, k1, p0, p1, s0, s1) \
    (((mode) == CUDNN_POOLING_MAX || (mode) == CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) ? miopenSet2dPoolingDescriptor(desc, mode, k0, k1, p0, p1, s0, s1) : miopenStatusNotImplemented)
#define cudnnPoolingBackward(h, p, a, y, d_y, dy, d_dy, x, d_x, b, dx, d_dx) \
    __cudnnPoolingBackward(h, p, a, y, d_y, dy, d_dy, x, d_x, b, dx, d_dx)
#define cudnnPoolingForward(h, p, a, x, d_x, b, y, d_y) \
    miopenPoolingForward(h, p, a, x, d_x, b, y, d_y, false, 0, 0)
    // __cudnnPoolingForward(h, p, a, x, d_x, b, y, d_y)
#define cudnnSetFilter4dDescriptor(desc, type, fmt, k, c, h, w) \
    cudnnSetTensor4dDescriptor(desc, fmt, type, k, c, h, w)


#define cudnnSoftmaxForward(hd, algo, mode, a, x, d_x, b, y, d_y) \
  ((mode) == CUDNN_SOFTMAX_MODE_CHANNEL ? miopenSoftmaxForward(hd, a, x, d_x, b, y, d_y) : ({ \
      miopenDataType_t dtype; int __dim[4], __stride[4]; \
      ensure_cmd(0 == cudnnGetTensor4dDescriptor(x, &dtype, &__dim[0], &__dim[1], &__dim[2], &__dim[3], &__stride[0], &__stride[1], &__stride[2], &__stride[3])); \
      cudnnTensorDescriptor_t desc; ensure_cmd(0 == cudnnCreateTensorDescriptor(&desc)); \
      ensure_cmd(0 == cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, __dim[0], __dim[1] * __dim[2] * __dim[3], 1, 1)); \
      miopenStatus_t st = miopenSoftmaxForward(hd, a, desc, d_x, b, desc, d_y); \
      ensure_cmd(miopenStatusSuccess == cudnnDestroyTensorDescriptor(desc)); \
      st; \
    }))

#define cudnnSoftmaxBackward(hd, algo, mode, a, y, d_y, dy, d_dy, b, dx, d_dx) \
  ((mode) == CUDNN_SOFTMAX_MODE_CHANNEL ? miopenSoftmaxBackward(hd, a, y, d_y, dy, d_dy, b, dx, d_dx) : ({ \
      miopenDataType_t dtype; int __dim[4], __stride[4]; \
      ensure_cmd(0 == cudnnGetTensor4dDescriptor(y, &dtype, &__dim[0], &__dim[1], &__dim[2], &__dim[3], &__stride[0], &__stride[1], &__stride[2], &__stride[3])); \
      cudnnTensorDescriptor_t desc; ensure_cmd(0 == cudnnCreateTensorDescriptor(&desc)); \
      ensure_cmd(0 == cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, __dim[0], __dim[1] * __dim[2] * __dim[3], 1, 1)); \
      miopenStatus_t st = miopenSoftmaxBackward(hd, a, desc, d_y, desc, d_dy, b, desc, d_dx); \
      ensure_cmd(miopenStatusSuccess == cudnnDestroyTensorDescriptor(desc)); \
      st; \
    }))

#define cudnnSetConvolutionGroupCount miopenSetConvolutionGroupCount

#define cudnnCreate miopenCreate
#define cudnnDestroy miopenDestroy
#define cudnnSetStream miopenSetStream
#define cudnnGetStream miopenGetStream
#define cudnnBatchNormMode_t miopenBatchNormMode_t
#define cudnnStatus_t miopenStatus_t

#define CUDNN_DATA_DOUBLE ((miopenDataType_t)-1)
#define CUDNN_DATA_HALF miopenHalf
#define CUDNN_BN_MIN_EPSILON 1e-5
#define CUDNN_BATCHNORM_PER_ACTIVATION miopenBNPerActivation
#define CUDNN_BATCHNORM_SPATIAL miopenBNSpatial
#define CUDNN_TENSOR_OP_MATH 0

#define cudnnAddTensor __cudnnAddTensor

#define cudnnConvolutionFwdAlgo_t miopenConvFwdAlgorithm_t
#define cudnnConvolutionBwdDataAlgo_t miopenConvBwdDataAlgorithm_t
#define cudnnConvolutionBwdFilterAlgo_t miopenConvBwdWeightsAlgorithm_t

#define cudnnGetConvolutionForwardAlgorithm(h, x, w, conv, y, pref, mlimit, algo) (*(algo) = miopenConvolutionFwdAlgoGEMM, miopenStatusSuccess)
#define cudnnGetConvolutionBackwardDataAlgorithm(h, w, dy, conv, dx, pref, mlimit, algo) (*(algo) = miopenConvolutionBwdDataAlgoGEMM, miopenStatusSuccess)
#define cudnnGetConvolutionBackwardFilterAlgorithm(h, x, dy, conv, dw, pref, mlimit, algo) (*(algo) = miopenConvolutionBwdWeightsAlgoGEMM, miopenStatusSuccess )

inline cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y) {
  return miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace, workSpaceSizeInBytes);
}


#define cudnnConvolutionBackwardData(hd, a, w, d_w, dy, d_dy, conv, algo, d_ws, wsize, b, dx, d_dx) \
  miopenConvolutionBackwardData(hd, a, dy, d_dy, w, d_w, conv, algo, b, dx, d_dx, d_ws, wsize)
#define cudnnConvolutionBackwardFilter(h, a, x, d_x, dy, d_dy, conv, algo, d_ws, wsize, b, dw, d_dw) \
         miopenConvolutionBackwardWeights(h, a, dy, d_dy, x, d_x, conv, algo, b, dw, d_dw, d_ws, wsize)

#define cudnnConvolutionBackwardBias miopenConvolutionBackwardBias
#define cudnnGetConvolution2dForwardOutputDim miopenGetConvolutionForwardOutputDim

#define CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST 0
#define CUDNN_CONVOLUTION_FWD_PREFER_FASTEST 0
#define CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST 0
#define cudnnMathType_t int
#define cudnnDeterminism_t int
#define CUDNN_SOFTMAX_ACCURATE 0
#define CUDNN_NOT_PROPAGATE_NAN 0

#define cudnnFindConvolutionForwardAlgorithmEx(hd, x, d_x, w, d_w, conv, y, d_y, reqCount, retCount, result, d_ws, wsize) ( \
          ensure_cmd((reqCount) >= 1), *(retCount) = 1, (result)->algo = miopenConvolutionFwdAlgoGEMM, \
          (result)->time = 1.0f, (result)->status = miopenStatusSuccess, (result)->determinism = 0, (result)->mathType = 0, \
          cudnnGetConvolutionForwardWorkspaceSize(hd, x, w, conv, y, (result)->algo, &(result)->memory))
        // miopenFindConvolutionForwardAlgorithm(hd, x, d_x, w, d_w, conv, y, d_y, reqCount, retCount, result, d_ws, wsize, false)

#define cudnnFindConvolutionBackwardDataAlgorithmEx(hd, w, d_w, dy, d_dy, conv, dx, d_dx, reqCount, retCount, result, d_ws, wsize) (ensure_cmd(0), miopenStatusSuccess)
        // miopenFindConvolutionBackwardDataAlgorithm(hd, w, d_w, dy, d_dy, conv, dx, d_dx, reqCount, retCount, result, d_ws, wsize, false)


struct cudnnConvolutionFwdAlgoPerf_t {
  cudnnConvolutionFwdAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];

  miopenConvAlgoPerf_t __internal;
};

struct cudnnConvolutionBwdDataAlgoPerf_t {
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];

  miopenConvAlgoPerf_t __internal;
};

#define WS_BOUND 8

inline cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t   handle,
    const   cudnnTensorDescriptor_t         xDesc,
    const   cudnnFilterDescriptor_t         wDesc,
    const   cudnnConvolutionDescriptor_t    convDesc,
    const   cudnnTensorDescriptor_t         yDesc,
    cudnnConvolutionFwdAlgo_t               algo,
    size_t                                 *sizeInBytes) {
  return miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc, sizeInBytes);
}

inline cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t                       handle,
    const cudnnFilterDescriptor_t       wDesc,
    const cudnnTensorDescriptor_t       dyDesc,
    const cudnnConvolutionDescriptor_t  convDesc,
    const cudnnTensorDescriptor_t       dxDesc,
    cudnnConvolutionBwdDataAlgo_t       algo,
    size_t                             *sizeInBytes) {

  if (algo != miopenConvolutionBwdDataAlgoGEMM)
    return miopenStatusNotImplemented;

  int stride[4], wshape[4], dxshape[4], dyshape[4];
  miopenDataType_t dtype;
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(wDesc, &dtype, wshape + 0, wshape + 1, wshape + 2, wshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(dxDesc, &dtype, dxshape + 0, dxshape + 1, dxshape + 2, dxshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(dyDesc, &dtype, dyshape + 0, dyshape + 1, dyshape + 2, dyshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);

  *sizeInBytes = wshape[1] * wshape[2] * wshape[3] * dyshape[2] * dyshape[3] * sizeof(float) * WS_BOUND;
  // size_t w2 = 0; miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc, convDesc, dxDesc, &w2); printf("%zd v.s. %zd\n", *sizeInBytes, w2);

  return CUDNN_STATUS_SUCCESS;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t                       handle,
    const cudnnTensorDescriptor_t       xDesc,
    const cudnnTensorDescriptor_t       dyDesc,
    const cudnnConvolutionDescriptor_t  convDesc,
    const cudnnFilterDescriptor_t       dwDesc,
    cudnnConvolutionBwdFilterAlgo_t     algo,
    size_t                             *sizeInBytes) {

  if (algo != miopenConvolutionBwdWeightsAlgoGEMM)
    return miopenStatusNotImplemented;

  int stride[4], dwshape[4], xshape[4], dyshape[4];
  miopenDataType_t dtype;
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(dwDesc, &dtype, dwshape + 0, dwshape + 1, dwshape + 2, dwshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(xDesc, &dtype, xshape + 0, xshape + 1, xshape + 2, xshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);
  ensure_cmd(miopenStatusSuccess == cudnnGetTensor4dDescriptor(dyDesc, &dtype, dyshape + 0, dyshape + 1, dyshape + 2, dyshape + 3, stride + 0, stride + 1, stride + 2, stride + 3) && dtype == miopenFloat);

  *sizeInBytes = dwshape[1] * dwshape[2] * dwshape[3] * dyshape[2] * dyshape[3] * sizeof(float) * WS_BOUND;
  // size_t w2 = 0; miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc, xDesc, convDesc, dwDesc, &w2); printf("%zd v.s. %zd\n", *sizeInBytes, w2);

  return CUDNN_STATUS_SUCCESS;
}

inline cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
  if (nbDims != 4)
    return miopenStatusNotImplemented;
  int base = 1;
  for (int i = nbDims - 1; i >= 0; --i) {
    if (strideA[i] != base)
      return miopenStatusNotImplemented;
    base *= dimA[i];
  }
  return cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, dataType, dimA[0], dimA[1], dimA[2], dimA[3]);
}

inline cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
  int stride[4];
  cudnnStatus_t ret = cudnnGetTensor4dDescriptor(filterDesc, dataType,
      filterDimA + 0, filterDimA + 1, filterDimA + 2, filterDimA + 3, stride + 0, stride + 1, stride + 2, stride + 3);
  if (ret != miopenStatusSuccess)
    return ret;
  *format = CUDNN_TENSOR_NCHW;
  *nbDims = 4;
  return miopenStatusSuccess;
}

inline cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
  if (nbDims != 4)
    return miopenStatusNotImplemented;
  if (format != CUDNN_TENSOR_NCHW)
    return miopenStatusNotImplemented;
  return cudnnSetTensor4dDescriptor(filterDesc, CUDNN_TENSOR_NCHW, dataType, filterDimA[0], filterDimA[1], filterDimA[2], filterDimA[3]);
}

inline cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnDataType_t dataType, /* image data type */
                             int n,                    /* number of inputs (batch size) */
                             int c,                    /* number of input feature maps */
                             int h,                    /* height of input section */
                             int w,                    /* width of input section */
                             int nStride,
                             int cStride,
                             int hStride,
                             int wStride) {
  if (wStride != 1)
    return miopenStatusNotImplemented;
  if (hStride != w)
    return miopenStatusNotImplemented;
  if (cStride != h * w)
    return miopenStatusNotImplemented;
  if (nStride != c * h * w)
    return miopenStatusNotImplemented;
	return cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
}

inline cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
                            const int windowDimA[], const int paddingA[], const int strideA[]) {
  if (nbDims != 2)
    return miopenStatusNotImplemented;

  return miopenSet2dPoolingDescriptor(poolingDesc, mode,
    windowDimA[0], windowDimA[1], paddingA[0], paddingA[1], strideA[0], strideA[1]);
}

inline cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
                                const int padA[], const int filterStrideA[], const int dilationA[],
                                cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  if (arrayLength != 2)
    return miopenStatusNotImplemented;
  return miopenInitConvolutionDescriptor(convDesc, mode,
    padA[0], padA[1], filterStrideA[0], filterStrideA[1], dilationA[0], dilationA[1]);
}


//////////////////////////////////////
// CUDNN_RNN: not tested
#define CUDNN_LINEAR_INPUT miopenRNNlinear
#define CUDNN_RNN_ALGO_STANDARD miopenRNNdefault
#define CUDNN_BIDIRECTIONAL miopenRNNbidirection
#define CUDNN_UNIDIRECTIONAL miopenRNNunidirection
#define CUDNN_LSTM miopenLSTM
#define CUDNN_GRU miopenGRU
#define cudnnDirectionMode_t miopenRNNDirectionMode_t

#define CUDNN_RNN_RELU miopenRNNRELU
#define CUDNN_RNN_TANH miopenRNNTANH
#define cudnnCreateRNNDescriptor miopenCreateRNNDescriptor
#define cudnnDestroyRNNDescriptor miopenDestroyRNNDescriptor
#define cudnnRNNDescriptor_t miopenRNNDescriptor_t
#define cudnnRNNMode_t miopenRNNMode_t
#define cudnnGetRNNWorkspaceSize miopenGetRNNWorkspaceSize

UN_IMPLEMENTED(cudnnSetRNNDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnRNNForwardInference, cudnnStatus_t)
UN_IMPLEMENTED(cudnnGetRNNLinLayerMatrixParams, cudnnStatus_t)
UN_IMPLEMENTED(cudnnGetRNNLinLayerBiasParams, cudnnStatus_t)


#if MIOPEN_VERSION_MINOR <= 8
#define cudnnTransformTensor(hd, a, x, d_x, b, y, d_y)  __cudnnTransformTensor(hd, a, x, d_x, b, y, d_y)
#else
#define cudnnTransformTensor(hd, a, x, d_x, b, y, d_y)  miopenTransformTensor(hd, a, x, d_x, b, y, d_y)
#endif

// LRN: MIOpen 1.7.1 not exactly mapped
#define CUDNN_LRN_MIN_N 1
#define CUDNN_LRN_MAX_N 16
#define CUDNN_LRN_MIN_K 1e-5
#define CUDNN_LRN_MIN_BETA 0.01

#define cudnnLRNDescriptor_t miopenLRNDescriptor_t
#define CUDNN_LRN_CROSS_CHANNEL_DIM1 miopenLRNCrossChannel
#define cudnnLRNMode_t miopenLRNMode_t
#define cudnnCreateLRNDescriptor miopenCreateLRNDescriptor
#define cudnnDestroyLRNDescriptor miopenDestroyLRNDescriptor
#define cudnnLRNCrossChannelForward(hd, lrn, mode, a, x, d_x, b, y, d_y) \
          miopenLRNForward(hd, lrn, a, x, d_x, b, y, d_y, false, NULL)
#define cudnnSetLRNDescriptor(d, n, a, b, k) \
          miopenSetLRNDescriptor(d, miopenLRNCrossChannel, n, a, b, k)
// #define cudnnLRNCrossChannelBackward(hd, lrn, mode, a, y, d_y, dy, d_dy, x, d_x, b, dx, d_dx) miopenLRNBackward(hd, lrn, a, y, d_y, dy, d_dy, x, d_x, b, dx, d_dx, NULL) // FIXME
UN_IMPLEMENTED(cudnnLRNCrossChannelBackward, cudnnStatus_t)


// Dropout: MIOpen 1.7.1 unsupported
#define cudnnDropoutDescriptor_t void*

UN_IMPLEMENTED(cudnnRestoreDropoutDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDropoutForward, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDropoutBackward, cudnnStatus_t)
UN_IMPLEMENTED(cudnnCreateDropoutDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDestroyDropoutDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnSetDropoutDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDropoutGetStatesSize, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDropoutGetReserveSpaceSize, cudnnStatus_t)

// Reduce: MIOpen 1.7.1 unsupported
#define cudnnReduceTensorOp_t int
#define cudnnReduceTensorIndices_t int
#define cudnnReduceTensorDescriptor_t void*

#define CUDNN_32BIT_INDICES 0
#define CUDNN_REDUCE_TENSOR_AMAX 0
#define CUDNN_REDUCE_TENSOR_AVG 0
#define CUDNN_REDUCE_TENSOR_NORM2 0
#define CUDNN_REDUCE_TENSOR_MAX 0
#define CUDNN_REDUCE_TENSOR_MIN 0
#define CUDNN_REDUCE_TENSOR_MUL 0
#define CUDNN_REDUCE_TENSOR_ADD 0
#define CUDNN_REDUCE_TENSOR_FLATTENED_INDICES 0
#define CUDNN_REDUCE_TENSOR_NORM1 0
#define CUDNN_REDUCE_TENSOR_NO_INDICES 0

UN_IMPLEMENTED(cudnnCreateReduceTensorDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnDestroyReduceTensorDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnGetReductionIndicesSize, cudnnStatus_t)
UN_IMPLEMENTED(cudnnSetReduceTensorDescriptor, cudnnStatus_t)
UN_IMPLEMENTED(cudnnGetReductionWorkspaceSize, cudnnStatus_t)
UN_IMPLEMENTED(cudnnReduceTensor, cudnnStatus_t)


/////////////////////////////////////////////////////////////////

template<class T> static inline T& getVariable(const void *key) {
  static std::unordered_map<const void*, void*>* __root = NULL;
  const char *global_symbol = "GLOBAL_VARS";

  if (__root == NULL) {
    const char *p_str_ptr = getenv(global_symbol);
    if (p_str_ptr == NULL) {
      char str_ptr[32];
      auto *__alter_root = new std::unordered_map<const void*, void*>();
      snprintf(str_ptr, sizeof(str_ptr), "%p", __alter_root);
      setenv(global_symbol, str_ptr, 0);

      // Avoid multi-thread pollution
      p_str_ptr = getenv(global_symbol);
      if (strcmp(p_str_ptr, str_ptr)) {
        delete __alter_root;
      }
    }
    sscanf(p_str_ptr, "%p", &__root);
    ensure_cmd(__root != NULL);
  }
  T* val = (T*)(*__root)[key];
  if (!val) {
    val = new T();
    (*__root)[key] = val;
  }
  return *val;
}

static inline hipError_t __managed_malloc(void **dptr, size_t bytes) {
    auto &it = getVariable<std::unordered_map<size_t, std::vector<void*>>>("__len2ptrs")[bytes];
    if (it.size()) {
      *dptr = it.back();
      it.pop_back();
      return hipSuccess;
    }
    ensure_cmd(hipSuccess == hipMalloc(dptr, bytes));
    getVariable<std::unordered_map<void*, size_t>>("__ptr2len")[*dptr] = bytes;
    return hipSuccess;
}

static inline hipError_t __managed_free(void *dptr) {
    if (!dptr)
      return hipDeviceSynchronize();

    size_t bytes = getVariable<std::unordered_map<void*, size_t>>("__ptr2len")[dptr];
    getVariable<std::unordered_map<size_t, std::vector<void*>>>("__len2ptrs")[bytes].push_back(dptr);
    return hipSuccess;

    // return hipFree(dptr);
}

#define hipMalloc __managed_malloc
#define hipFree __managed_free

#endif // __HIP_RUNTIME_H__
