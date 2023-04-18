#include <stdexcept>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cudnn.h>
#include "nnfusion_rt.h"
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif

__global__ void printTensor(float* data, int n) {
    for (int i = 0; i < min(n, 10); i++) printf("%f ", data[i]);
    if (n > 10) printf("... %f", data[n-1]);
}

#define DEBUG_TENSOR(tensor, size) { \
    printf("%s: ", #tensor); \
    printTensor<<<1, 1>>>(tensor, size); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
    fflush(stdout); \
    printf("\n"); \
}

__global__ void printTensorChar(char* data, int n) {
    for (int i = 0; i < min(n, 10); i++) printf("%d ", (int) data[i]);
    if (n > 10) printf("... %d", (int) data[n-1]);
}

#define DEBUG_TENSOR_CHAR(tensor, size) { \
    printf("%s: ", #tensor); \
    printTensorChar<<<1, 1>>>(tensor, size); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
    fflush(stdout); \
    printf("\n"); \
}

__global__ void printTensor3D(float* data, int n, int c, int stride) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            printf("%f ", data[(j * n + i) * stride]);
        }
        printf("\n");
    }
    // if (n > 10) printf("... %f", data[n-1]);
}

#define DEBUG_TENSOR1(tensor, n, c, stride) { \
    printf("%s: ", #tensor); \
    printTensor3D<<<1, 1>>>(tensor, n, c, stride); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
    fflush(stdout); \
    printf("\n"); \
}
#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
   
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_6_0;
float* Reshape_7_0;
float* v_0;
float* Result_16_0;
float* Constant_2_0;
float* Reshape_3_0;
float* k_0;
float* Result_15_0;
int64_t* gen_id_0;
int64_t* Result_14_0;
// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN
int get_device_type()
{
    return 0;
}
// Node name:	Result_16
// Description:	Result
// Input:
//	- name: v_0	type: float	shape: Shape{1, 12, 64, 64}
// Output:
//	- name: Result_16_0	type: float	shape: Shape{1, 12, 64, 64}
void Result_float_float_cuda_lib_Result_16(cudaStream_t stream, float* input0, float* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,196608, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Reshape_7_0	type: float	shape: Shape{1}
//	- name: Parameter_1_0	type: float	shape: Shape{1, 12, 64, 64}
// Output:
//	- name: v_0	type: float	shape: Shape{1, 12, 64, 64}
// Fused functions:
// Broadcast, Broadcast_8
// Add, v_0
extern "C" __launch_bounds__(64) __global__ void FusedKernel_float_float_float_cuda_Broadcast_Add_1(float* input0, float* input1, float* output0)
{
    int tid = blockIdx.x * 64 + threadIdx.x;
    float temp0 = input0[tid % 1];
    float temp1 = add(input1[tid], temp0);
    output0[tid] = temp1;

}
extern void FusedKernel_float_float_float_cuda_Broadcast_Add_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    FusedKernel_float_float_float_cuda_Broadcast_Add_1<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: gen_id_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_11(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/gen_id_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load gen_id_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Result_14
// Description:	Result
// Input:
//	- name: gen_id_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Result_14_0	type: int64_t	shape: Shape{}
void Result_int64_t_int64_t_cuda_lib_Result_14(cudaStream_t stream, int64_t* input0, int64_t* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,8, cudaMemcpyDeviceToDevice, stream));
}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 2
#define NNFUSION_GRAPH_OUTPUT_NUM 3
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {1, 12, 64, 64}
#define NNFUSION_GRAPH_INPUT_DTYPE_1 float
#define NNFUSION_GRAPH_INPUT_SHAPE_1 {1, 12, 64, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 int64_t
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_1 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_1 {1, 12, 64, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_2 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_2 {1, 12, 64, 64}
#endif

// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[4];
    bin_file.read(tmp_mem, 4);
    cudaMemcpyAsync(output0, tmp_mem, 4, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}

extern "C" void cuda_init()
{
//CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:393408
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,393408));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 393408));
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Reshape_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
v_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
Result_16_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196672);
Reshape_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196672);
k_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196736);
Result_15_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196736);
gen_id_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+393344);
Result_14_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+393344);
// create streams/handles
 // name=@tmp_1
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=@tmp_0
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=gen_id_0
Constant_int64_t_cuda_Constant_11(0, Result_14_0);
}


extern "C" int kernel_entry(float* Parameter_0_0, float* Parameter_1_0, int64_t* Result_14_0, float* Result_15_0, float* Result_16_0)
{
// kernel_entry_init
 // name=Reshape_7
// eliminated: Reshape_float_float_cuda_Reshape_7_Call(dim3(1, 1, 1), dim3(64, 1, 1), 0, 0, Constant_6_0, Reshape_7_0);
 // name=ElementWiseFused_18
FusedKernel_float_float_float_cuda_Broadcast_Add_1_Call(dim3(768, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_7_0, Parameter_1_0, Result_16_0);
 // name=Result_16
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_16(0, Result_16_0, Result_16_0);
 // name=Reshape_3
// eliminated: Reshape_float_float_cuda_Reshape_3_Call(dim3(1, 1, 1), dim3(64, 1, 1), 0, 0, Constant_2_0, Reshape_3_0);
 // name=ElementWiseFused_17
FusedKernel_float_float_float_cuda_Broadcast_Add_1_Call(dim3(768, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_3_0, Parameter_0_0, Result_15_0);
 // name=Result_15
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_16(0, Result_15_0, Result_15_0);
 // name=Result_14
Result_int64_t_int64_t_cuda_lib_Result_14(0, gen_id_0, Result_14_0);
return 0;
}


extern "C" void cuda_free()
{
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

