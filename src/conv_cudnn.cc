#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDNN_CALL(cmd)                                                        \
    do {                                                                       \
        cudnnStatus_t e = cmd;                                                 \
        if (e != CUDNN_STATUS_SUCCESS) {                                       \
            printf("Failed: Cudnn error %s:%d '%s'\n", __FILE__, __LINE__,     \
                   cudnnGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static cudnnHandle_t cudnn_handle;
static bool initialized = false;
static void* work_data = nullptr;

int CuDNN_DLGpuConv2d(float * x, float * w, float * output) {
    size_t input_N = 1;
    size_t input_C = 64;
    size_t input_H = 64;
    size_t input_W = 64;
    const float* input_data = (const float*)x;
    if (!initialized) {
        cudnnCreate(&cudnn_handle);
        initialized = true;
    }

    // input
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, input_N, input_C,
        input_H, input_W));
    size_t filter_N = 64;
    size_t filter_C = 64;
    size_t filter_H = 1;
    size_t filter_W = 1;
    const float* filter_data = (const float*)w;

    // filter
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, filter_N, filter_C,
        filter_H, filter_W));

    // convolution
    int padding_h = (filter_H - 1) / 2, padding_w = (filter_W - 1) / 2, stride_h = 1, stride_w = 1;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    size_t out_N = 1;
    size_t out_C = 64;
    size_t out_H = 64;
    size_t out_W = 64;
    // output
    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, out_N, out_C, out_H,
        out_W));
    float* output_data = (float*)output;
    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    // algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle, input_desc, filter_desc, conv_desc, out_desc, algo,
        &workspace_size));
    if (work_data == nullptr)
        cudaMalloc(&work_data, workspace_size);

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn_handle, &alpha, input_desc, input_data, filter_desc,
        filter_data, conv_desc, algo, work_data, workspace_size, &beta,
        out_desc, output_data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    return 0;
}


extern "C" {
void conv_cudnn(float *x, float *w, float *output) {
    CuDNN_DLGpuConv2d(x, w, output);
    // default_function_kernel1<<<dim3(8, 8, 1), dim3(4, 4, 8)>>>(x, w, output);
    // default_function_kernel0<<<64, 64>>>(output, x, w);
}
}
