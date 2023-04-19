#include <stdexcept>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <sstream>
#include "nnfusion_rt.h"
#include <fstream>
// #include <cudnn.h>

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
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif
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
__device__ __forceinline__ float mul(float x0, float x1)
{
    return x0 * x1;
}
__device__ __forceinline__ char  load(const char*  __restrict__ in, int i=0, bool b=true)
{
    char v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ float  load(const float*  __restrict__ in, int i=0, bool b=true)
{
    float v = 0.0f;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int32_t  load(const int32_t*  __restrict__ in, int i=0, bool b=true)
{
    int32_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t  load(const int64_t*  __restrict__ in, int i=0, bool b=true)
{
    int64_t v = 0;
    if (b)
    {
        v = __ldg(in + i);
    }
    return v;
}
__device__ __forceinline__ int64_t add(int64_t x0, int64_t x1)
{
    return x0 + x1;
}
__device__ __forceinline__ int division_by_invariant_multiplication(int value, int magic, int shift)
{
    int result;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        ".reg .u64 res64;\n\t"
        ".reg .u32 lo32, hi32;\n\t"
        "setp.ne.s32 p, %2, 1;\n\t"
        "mul.wide.u32 res64, %1, %2;\n\t"
        "mov.b64 {lo32, hi32}, res64;\n\t"
        "selp.u32 hi32, hi32, %1, p;\n\t"
        "shr.u32 %0, hi32, %3;\n\t"
        "}" : "=r"(result) : "r"(value), "r"(magic), "r"(shift));
    return result;
}
__device__ __forceinline__ int64_t select(char x0, int64_t x1, int64_t x2)
{
    return (x0 == 0) ? x2 : x1;
}

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* Reshape_12_0;
float* Broadcast_40_0;
float* BatchMatMul_24_0;
float* BatchMatMul_10_0;
float* BatchMatMul_14_0;
float* Reshape_25_0;
float* Reshape_27_0;
int64_t* Select_30_0;
int64_t* Select_20_0;
float* Reshape_15_0;
float* Reshape_17_0;
float* Reshape_11_0;
float* Reshape_32_0;
float* Reshape_34_0;
float* BatchMatMul_35_0;
float* Reshape_36_0;
float* Reshape_37_0;
float* Multiply_41_0;
float* Softmax_42_0;
float* Reshape_43_0;
float* BatchMatMul_45_0;
float* Reshape_46_0;
float* Reshape_47_0;
float* BatchMatMul_49_0;
float* x_2;
float* Result_55_0;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
float* Constant_38_0;
float* Reshape_39_0;
float* Constant_2_0;
float* Reshape_23_0;
float* Constant_0_0;
float* Reshape_9_0;
float* Constant_1_0;
float* Reshape_13_0;
int64_t* Constant_51_0;
int64_t* Constant_28_0;
char* Constant_29_0;
int64_t* Constant_18_0;
char* Constant_19_0;
int64_t* gen_id_1;
float* tensor_31;
float* tensor_21;
float* Result_56_0;
float* Constant_3_0;
float* Reshape_48_0;
float* Reshape_44_0;
float* Reshape_33_0;
float* Result_54_0;
int64_t* Result_53_0;
// Node name:	Constant_1
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1_0	type: float	shape: Shape{12, 64, 64}
void Constant_float_cuda_Constant_1(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[196608];
    bin_file.read(tmp_mem, 196608);
    cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Constant_19_0	type: char	shape: Shape{64, 12, 3}
//	- name: Constant_18_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Parameter_4_0	type: int64_t	shape: Shape{}
//	- name: Constant_29_0	type: char	shape: Shape{64, 12, 3}
//	- name: Constant_28_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Constant_51_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Select_20_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Select_30_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: gen_id_1	type: int64_t	shape: Shape{}
// Fused functions:
// Select_char_int64_t_int64_t_int64_t_cuda_Select_20<<<dim3(18, 1, 1), dim3(128, 1, 1), 0, 0>>>(Constant_19_0, Constant_18_0, Parameter_4_0, Select_20_0);
// Select_char_int64_t_int64_t_int64_t_cuda_Select_30<<<dim3(18, 1, 1), dim3(128, 1, 1), 0, 0>>>(Constant_29_0, Constant_28_0, Parameter_4_0, Select_30_0);
// Add_int64_t_int64_t_int64_t_cuda_Add_52<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, 0>>>(Parameter_4_0, Constant_51_0, gen_id_1);
// Deduped function map: <src_function_name : deduped_function_name>
// Select_char_int64_t_int64_t_int64_t_cuda_Select_30 : Select_char_int64_t_int64_t_int64_t_cuda_Select_20

// Node name:	Select_20
// Description:	Select
// Input:
//	- name: Constant_19_0	type: char	shape: Shape{64, 12, 3}
//	- name: Constant_18_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Parameter_4_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Select_20_0	type: int64_t	shape: Shape{64, 12, 3}
__device__ __noinline__ void Select_char_int64_t_int64_t_int64_t_cuda_Select_20_block_kernel(char* input0, int64_t* input1, int64_t* input2, int64_t* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(18, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[blockIdx.x * 128 + threadIdx.x] = select(input0[blockIdx.x * 128 + threadIdx.x], input1[blockIdx.x * 128 + threadIdx.x], input2[0]);

}
// Node name:	Add_52
// Description:	Add
// Input:
//	- name: Parameter_4_0	type: int64_t	shape: Shape{}
//	- name: Constant_51_0	type: int64_t	shape: Shape{}
// Output:
//	- name: gen_id_1	type: int64_t	shape: Shape{}
__device__ __noinline__ void Add_int64_t_int64_t_int64_t_cuda_Add_52_block_kernel(int64_t* input0, int64_t* input1, int64_t* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 1){
        return;
    }
    const dim3 blockDim(1, 1, 1);
    const dim3 gridDim(1, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[threadIdx.x] = add(input0[0], input1[0]);

}
extern "C" __global__  void BlockFusionKernel_char_int64_t_int64_t_char_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_Select_Select_Add_0(char* input0, int64_t* input1, int64_t* input2, char* input3, int64_t* input4, int64_t* input5, int64_t* output0, int64_t* output1, int64_t* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 17)
    {
        Select_char_int64_t_int64_t_int64_t_cuda_Select_20_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 18 && (int)blockIdx.x <= 35)
    {
        Select_char_int64_t_int64_t_int64_t_cuda_Select_20_block_kernel(input3, input4, input2, output1, threadIdx.x, blockIdx.x - 18 + 0, NULL);
    }
    else if (blockIdx.x == 36)
    {
        Add_int64_t_int64_t_int64_t_cuda_Add_52_block_kernel(input2, input5, output2, threadIdx.x, 0, NULL);
    }

}
extern void BlockFusionKernel_char_int64_t_int64_t_char_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_Select_Select_Add_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, char* input0, int64_t* input1, int64_t* input2, char* input3, int64_t* input4, int64_t* input5, int64_t* output0, int64_t* output1, int64_t* output2) {
    BlockFusionKernel_char_int64_t_int64_t_char_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_Select_Select_Add_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1, output2);
}
// Node name:	Result_53
// Description:	Result
// Input:
//	- name: gen_id_1	type: int64_t	shape: Shape{}
// Output:
//	- name: Result_53_0	type: int64_t	shape: Shape{}
void Result_int64_t_int64_t_cuda_lib_Result_53(cudaStream_t stream, int64_t* input0, int64_t* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,8, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	 BlockFusion
// Input:
//	- name: Parameter_7_0	type: float	shape: Shape{64, 12, 64, 64}
//	- name: Select_20_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Reshape_17_0	type: float	shape: Shape{64, 12, 64}
//	- name: Parameter_5_0	type: float	shape: Shape{64, 12, 64, 64}
//	- name: Select_30_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Reshape_27_0	type: float	shape: Shape{64, 12, 64}
// Output:
//	- name: tensor_21	type: float	shape: Shape{64, 12, 64, 64}
//	- name: tensor_31	type: float	shape: Shape{64, 12, 64, 64}
// Fused functions:
// ScatterND_float_int64_t_float_float_cuda_ScatterND_21<<<dim3(384, 1, 1), dim3(128, 1, 1), 0, 0>>>(Parameter_7_0, Select_20_0, Reshape_17_0, tensor_21);
// ScatterND_float_int64_t_float_float_cuda_ScatterND_31<<<dim3(384, 1, 1), dim3(128, 1, 1), 0, 0>>>(Parameter_5_0, Select_30_0, Reshape_27_0, tensor_31);
// Deduped function map: <src_function_name : deduped_function_name>
// ScatterND_float_int64_t_float_float_cuda_ScatterND_31 : ScatterND_float_int64_t_float_float_cuda_ScatterND_21

// Node name:	ScatterND_21
// Description:	ScatterND
// Input:
//	- name: Parameter_7_0	type: float	shape: Shape{64, 12, 64, 64}
//	- name: Select_20_0	type: int64_t	shape: Shape{64, 12, 3}
//	- name: Reshape_17_0	type: float	shape: Shape{64, 12, 64}
// Output:
//	- name: tensor_21	type: float	shape: Shape{64, 12, 64, 64}
__device__ __noinline__ void ScatterND_float_int64_t_float_float_cuda_ScatterND_21_block_kernel(float* input0, int64_t* input1, float* input2, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(384, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 49152) { return; }
    uint32_t out_id = tid / 64;
    uint32_t in_id = tid % 64;
    uint32_t idx = in_id + input1[out_id * 3 + 0] * 49152 + input1[out_id * 3 + 1] * 4096 + input1[out_id * 3 + 2] * 64;
    input0[idx] = input2[tid];

}
extern "C" __global__  void BlockFusionKernel_float_int64_t_float_float_int64_t_float_float_float_cuda_ScatterND_ScatterND_2(float* input0, int64_t* input1, float* input2, float* input3, int64_t* input4, float* input5, float* output0, float* output1)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 383)
    {
        ScatterND_float_int64_t_float_float_cuda_ScatterND_21_block_kernel(input0, input1, input2, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 767)
    {
        ScatterND_float_int64_t_float_float_cuda_ScatterND_21_block_kernel(input3, input4, input5, output1, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_int64_t_float_float_int64_t_float_float_float_cuda_ScatterND_ScatterND_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int64_t* input1, float* input2, float* input3, int64_t* input4, float* input5, float* output0, float* output1) {
    BlockFusionKernel_float_int64_t_float_float_int64_t_float_float_float_cuda_ScatterND_ScatterND_2<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, output0, output1);
}
// Node name:	BatchMatMul_45
// Description:	BatchMatMul
// Input:
//	- name: Reshape_43_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Reshape_44_0	type: float	shape: Shape{64, 12, 64, 64}
// Output:
//	- name: BatchMatMul_45_0	type: float	shape: Shape{64, 12, 1, 64}
extern "C" __global__ void BatchMatMul_float_float_float_cuda_BatchMatMul_45(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute)
{
    __shared__ float A_shared[64];
    __shared__ float B_shared[4096];
    {
        {
          float compute_local[2];
          
          
          float A_shared_local[1];
          float B_shared_local[2];
          compute_local[(0)] = 0.000000e+00f;
          compute_local[(1)] = 0.000000e+00f;
          for (int k_outer = 0; k_outer < 2; ++k_outer) {
            __syncthreads();
            A_shared[(((int)threadIdx.x))] = A[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[(((int)threadIdx.x))] = B[((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)))];
            B_shared[((((int)threadIdx.x) + 64))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 64))];
            B_shared[((((int)threadIdx.x) + 128))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 128))];
            B_shared[((((int)threadIdx.x) + 192))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 192))];
            B_shared[((((int)threadIdx.x) + 256))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 256))];
            B_shared[((((int)threadIdx.x) + 320))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 320))];
            B_shared[((((int)threadIdx.x) + 384))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 384))];
            B_shared[((((int)threadIdx.x) + 448))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 448))];
            B_shared[((((int)threadIdx.x) + 512))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 512))];
            B_shared[((((int)threadIdx.x) + 576))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 576))];
            B_shared[((((int)threadIdx.x) + 640))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 640))];
            B_shared[((((int)threadIdx.x) + 704))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 704))];
            B_shared[((((int)threadIdx.x) + 768))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 768))];
            B_shared[((((int)threadIdx.x) + 832))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 832))];
            B_shared[((((int)threadIdx.x) + 896))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 896))];
            B_shared[((((int)threadIdx.x) + 960))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 960))];
            B_shared[((((int)threadIdx.x) + 1024))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1024))];
            B_shared[((((int)threadIdx.x) + 1088))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1088))];
            B_shared[((((int)threadIdx.x) + 1152))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1152))];
            B_shared[((((int)threadIdx.x) + 1216))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1216))];
            B_shared[((((int)threadIdx.x) + 1280))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1280))];
            B_shared[((((int)threadIdx.x) + 1344))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1344))];
            B_shared[((((int)threadIdx.x) + 1408))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1408))];
            B_shared[((((int)threadIdx.x) + 1472))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1472))];
            B_shared[((((int)threadIdx.x) + 1536))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1536))];
            B_shared[((((int)threadIdx.x) + 1600))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1600))];
            B_shared[((((int)threadIdx.x) + 1664))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1664))];
            B_shared[((((int)threadIdx.x) + 1728))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1728))];
            B_shared[((((int)threadIdx.x) + 1792))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1792))];
            B_shared[((((int)threadIdx.x) + 1856))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1856))];
            B_shared[((((int)threadIdx.x) + 1920))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1920))];
            B_shared[((((int)threadIdx.x) + 1984))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 1984))];
            B_shared[((((int)threadIdx.x) + 2048))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4096))];
            B_shared[((((int)threadIdx.x) + 2112))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4160))];
            B_shared[((((int)threadIdx.x) + 2176))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4224))];
            B_shared[((((int)threadIdx.x) + 2240))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4288))];
            B_shared[((((int)threadIdx.x) + 2304))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4352))];
            B_shared[((((int)threadIdx.x) + 2368))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4416))];
            B_shared[((((int)threadIdx.x) + 2432))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4480))];
            B_shared[((((int)threadIdx.x) + 2496))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4544))];
            B_shared[((((int)threadIdx.x) + 2560))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4608))];
            B_shared[((((int)threadIdx.x) + 2624))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4672))];
            B_shared[((((int)threadIdx.x) + 2688))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4736))];
            B_shared[((((int)threadIdx.x) + 2752))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4800))];
            B_shared[((((int)threadIdx.x) + 2816))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4864))];
            B_shared[((((int)threadIdx.x) + 2880))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4928))];
            B_shared[((((int)threadIdx.x) + 2944))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 4992))];
            B_shared[((((int)threadIdx.x) + 3008))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5056))];
            B_shared[((((int)threadIdx.x) + 3072))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5120))];
            B_shared[((((int)threadIdx.x) + 3136))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5184))];
            B_shared[((((int)threadIdx.x) + 3200))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5248))];
            B_shared[((((int)threadIdx.x) + 3264))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5312))];
            B_shared[((((int)threadIdx.x) + 3328))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5376))];
            B_shared[((((int)threadIdx.x) + 3392))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5440))];
            B_shared[((((int)threadIdx.x) + 3456))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5504))];
            B_shared[((((int)threadIdx.x) + 3520))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5568))];
            B_shared[((((int)threadIdx.x) + 3584))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5632))];
            B_shared[((((int)threadIdx.x) + 3648))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5696))];
            B_shared[((((int)threadIdx.x) + 3712))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5760))];
            B_shared[((((int)threadIdx.x) + 3776))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5824))];
            B_shared[((((int)threadIdx.x) + 3840))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5888))];
            B_shared[((((int)threadIdx.x) + 3904))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 5952))];
            B_shared[((((int)threadIdx.x) + 3968))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 6016))];
            B_shared[((((int)threadIdx.x) + 4032))] = B[(((((((int)blockIdx.x) * 8192) + (k_outer * 2048)) + ((int)threadIdx.x)) + 6080))];
            __syncthreads();
            for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
              A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 5) * 32) + k_inner_outer))];
              B_shared_local[(0)] = B_shared[(((((((int)threadIdx.x) >> 5) * 2048) + (k_inner_outer * 64)) + (((int)threadIdx.x) & 31)))];
              B_shared_local[(1)] = B_shared[((((((((int)threadIdx.x) >> 5) * 2048) + (k_inner_outer * 64)) + (((int)threadIdx.x) & 31)) + 32))];
              compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
              compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
            }
          }
          compute[((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)))] = compute_local[(0)];
          compute[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32))] = compute_local[(1)];
        }


    }

}
extern void BatchMatMul_float_float_float_cuda_BatchMatMul_45_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
    BatchMatMul_float_float_float_cuda_BatchMatMul_45<<<grids, blocks, mem, stream>>>(A, B, compute);
}
// Node name:	BatchMatMul_35
// Description:	BatchMatMul
// Input:
//	- name: Reshape_33_0	type: float	shape: Shape{64, 12, 64, 64}
//	- name: Reshape_34_0	type: float	shape: Shape{64, 12, 64, 1}
// Output:
//	- name: BatchMatMul_35_0	type: float	shape: Shape{64, 12, 64, 1}
extern "C" __global__ void BatchMatMul_float_float_float_cuda_BatchMatMul_35(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute)
{
    __shared__ float A_shared[2048];
    __shared__ float B_shared[32];
    {
        {
          float compute_local[1];
          
          
          compute_local[(0)] = 0.000000e+00f;
          for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
            __syncthreads();
            ((float4*)(A_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(A + (((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
            ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 2048))))[0];
            ((float4*)(A_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 4096))))[0];
            ((float4*)(A_shared + (((((((((int)threadIdx.x) * 4) + 1536) >> 10) * 1024) + (((((int)threadIdx.x) >> 2) + 32) * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0] = ((float4*)(A + ((((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 1536) >> 10) * 4096)) + (((((int)threadIdx.x) >> 2) + 32) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
            if (((int)threadIdx.x) < 8) {
              B_shared[((((int)threadIdx.x) * 4))] = B[(((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))];
            }
            if (((int)threadIdx.x) < 8) {
              B_shared[(((((int)threadIdx.x) * 4) + 1))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 1) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 1) & 15)))];
            }
            if (((int)threadIdx.x) < 8) {
              B_shared[(((((int)threadIdx.x) * 4) + 2))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 2) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 2) & 15)))];
            }
            if (((int)threadIdx.x) < 8) {
              B_shared[(((((int)threadIdx.x) * 4) + 3))] = B[(((((((int)blockIdx.x) * 128) + ((((((int)threadIdx.x) * 4) + 3) >> 4) * 64)) + (k_outer_outer * 16)) + (((((int)threadIdx.x) * 4) + 3) & 15)))];
            }
            __syncthreads();
            compute_local[(0)] = (compute_local[(0)] + (A_shared[((((int)threadIdx.x) * 16))] * B_shared[(((((int)threadIdx.x) >> 6) * 16))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 1))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 1))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 2))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 2))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 3))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 3))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 4))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 4))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 5))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 5))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 6))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 6))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 7))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 7))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 8))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 8))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 9))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 9))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 10))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 10))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 11))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 11))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 12))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 12))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 13))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 13))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 14))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 14))]));
            compute_local[(0)] = (compute_local[(0)] + (A_shared[(((((int)threadIdx.x) * 16) + 15))] * B_shared[((((((int)threadIdx.x) >> 6) * 16) + 15))]));
          }
          compute[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))] = compute_local[(0)];
        }


    }

}
extern void BatchMatMul_float_float_float_cuda_BatchMatMul_35_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
    BatchMatMul_float_float_float_cuda_BatchMatMul_35<<<grids, blocks, mem, stream>>>(A, B, compute);
}
// Node name:	Reshape_44
// Description:	Reshape
// Input:
//	- name: tensor_31	type: float	shape: Shape{64, 12, 64, 64}
// Output:
//	- name: Reshape_44_0	type: float	shape: Shape{64, 12, 64, 64}
extern "C" __launch_bounds__(128) __global__ void Reshape_float_float_cuda_Reshape_44(float* input0, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 3145728) { return; }
    output0[tid] = input0[tid];

}
extern void Reshape_float_float_cuda_Reshape_44_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_44<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Multiply_41
// Description:	Multiply
// Input:
//	- name: Reshape_37_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Broadcast_40_0	type: float	shape: Shape{64, 12, 1, 64}
// Output:
//	- name: Multiply_41_0	type: float	shape: Shape{64, 12, 1, 64}
extern "C" __launch_bounds__(128) __global__ void Multiply_float_float_float_cuda_Multiply_41(float* input0, float* input1, float* output0)
{
    output0[blockIdx.x * 128 + threadIdx.x] = mul(input0[blockIdx.x * 128 + threadIdx.x], input1[blockIdx.x * 128 + threadIdx.x]);

}
extern void Multiply_float_float_float_cuda_Multiply_41_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* output0) {
    Multiply_float_float_float_cuda_Multiply_41<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Result_55
// Description:	Result
// Input:
//	- name: x_2	type: float	shape: Shape{64, 12, 1, 64}
// Output:
//	- name: Result_55_0	type: float	shape: Shape{64, 12, 1, 64}
void Result_float_float_cuda_lib_Result_55(cudaStream_t stream, float* input0, float* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,196608, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	Reshape_12
// Description:	Reshape
// Input:
//	- name: Parameter_6_0	type: float	shape: Shape{64, 12, 1, 64}
// Output:
//	- name: Reshape_12_0	type: float	shape: Shape{64, 12, 1, 64}
extern "C" __launch_bounds__(128) __global__ void Reshape_float_float_cuda_Reshape_12(float* input0, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 49152) { return; }
    output0[tid] = input0[tid];

}
extern void Reshape_float_float_cuda_Reshape_12_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_12<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Softmax_42
// Description:	Softmax
// Input:
//	- name: Multiply_41_0	type: float	shape: Shape{64, 12, 1, 64}
// Output:
//	- name: Softmax_42_0	type: float	shape: Shape{64, 12, 1, 64}
extern "C" __launch_bounds__(128) __global__ void Softmax_float_float_cuda_Softmax_42(float* input0, float* output0)
{

    int wrap_id = blockIdx.x * 4 + (threadIdx.x >> 5); 
    if (wrap_id >= 768) {
        return;
    }
    int lane_id = threadIdx.x & 31;
    float local[2];
    for (int i = 0; i < 2; i++){
        local[i] = input0[wrap_id * 64 + i * 32 + lane_id];
    }
    float max_value = local[0];
    for (int i = 1; i < 2; i++){
        max_value = max(max_value, local[i]);
    }
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));

    float sum = 0;
    for (int i = 0; i < 2; i++){
        local[i] = expf(local[i] - max_value);
        sum += local[i];
    }

    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    for (int i = 0; i < 2; i++){
        output0[wrap_id * 64 + i * 32 + lane_id] = local[i] / sum;
    }

}
extern void Softmax_float_float_cuda_Softmax_42_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Softmax_float_float_cuda_Softmax_42<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Result_56
// Description:	Result
// Input:
//	- name: tensor_21	type: float	shape: Shape{64, 12, 64, 64}
// Output:
//	- name: Result_56_0	type: float	shape: Shape{64, 12, 64, 64}
void Result_float_float_cuda_lib_Result_56(cudaStream_t stream, float* input0, float* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,12582912, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	Constant_19
// Description:	Constant
// Input:
// Output:
//	- name: Constant_19_0	type: char	shape: Shape{64, 12, 3}
void Constant_char_cuda_Constant_19(cudaStream_t stream, char* output0)
{
    std::ifstream bin_file("./Constant/Constant_19_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_19_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2_0	type: float	shape: Shape{12, 64, 64}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[196608];
    bin_file.read(tmp_mem, 196608);
    cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_28
// Description:	Constant
// Input:
// Output:
//	- name: Constant_28_0	type: int64_t	shape: Shape{64, 12, 3}
void Constant_int64_t_cuda_Constant_28(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_28_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_28_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[18432];
    bin_file.read(tmp_mem, 18432);
    cudaMemcpyAsync(output0, tmp_mem, 18432, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3_0	type: float	shape: Shape{12, 64, 64}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[196608];
    bin_file.read(tmp_mem, 196608);
    cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_51
// Description:	Constant
// Input:
// Output:
//	- name: Constant_51_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_51(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_51_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_51_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_0
// Description:	Constant
// Input:
// Output:
//	- name: Constant_0_0	type: float	shape: Shape{12, 64, 64}
void Constant_float_cuda_Constant_0(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_0_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_0_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[196608];
    bin_file.read(tmp_mem, 196608);
    cudaMemcpyAsync(output0, tmp_mem, 196608, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	BatchMatMul_49
// Description:	BatchMatMul
// Input:
//	- name: Reshape_47_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Reshape_48_0	type: float	shape: Shape{12, 64, 64}
// Output:
//	- name: BatchMatMul_49_0	type: float	shape: Shape{64, 12, 1, 64}
extern "C" __global__ void BatchMatMul_float_float_float_cuda_BatchMatMul_49(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute)
{
    __shared__ float A_shared[1024];
    __shared__ float B_shared[1024];
    {
        {
          float compute_local[2];
          
          
          float A_shared_local[1];
          float B_shared_local[2];
          compute_local[(0)] = 0.000000e+00f;
          compute_local[(1)] = 0.000000e+00f;
          A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)))];
          A_shared[((((int)threadIdx.x) + 128))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 1536))];
          A_shared[((((int)threadIdx.x) + 256))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 3072))];
          A_shared[((((int)threadIdx.x) + 384))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 4608))];
          A_shared[((((int)threadIdx.x) + 512))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 6144))];
          A_shared[((((int)threadIdx.x) + 640))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 7680))];
          A_shared[((((int)threadIdx.x) + 768))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 9216))];
          A_shared[((((int)threadIdx.x) + 896))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 10752))];
          B_shared[(((int)threadIdx.x))] = B[(((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))];
          B_shared[((((int)threadIdx.x) + 128))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 512))];
          B_shared[((((int)threadIdx.x) + 256))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1024))];
          B_shared[((((int)threadIdx.x) + 384))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1536))];
          B_shared[((((int)threadIdx.x) + 512))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2048))];
          B_shared[((((int)threadIdx.x) + 640))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2560))];
          B_shared[((((int)threadIdx.x) + 768))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3072))];
          B_shared[((((int)threadIdx.x) + 896))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3584))];
          __syncthreads();
          for (int k_inner_outer = 0; k_inner_outer < 64; ++k_inner_outer) {
            A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 3) * 64) + k_inner_outer))];
            B_shared_local[(0)] = B_shared[(((k_inner_outer * 16) + (((int)threadIdx.x) & 7)))];
            B_shared_local[(1)] = B_shared[((((k_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8))];
            compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
            compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
          }
          compute[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)))] = compute_local[(0)];
          compute[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)) + 8))] = compute_local[(1)];
        }


    }

}
extern void BatchMatMul_float_float_float_cuda_BatchMatMul_49_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
    BatchMatMul_float_float_float_cuda_BatchMatMul_49<<<grids, blocks, mem, stream>>>(A, B, compute);
}
// Node name:	Constant_29
// Description:	Constant
// Input:
// Output:
//	- name: Constant_29_0	type: char	shape: Shape{64, 12, 3}
void Constant_char_cuda_Constant_29(cudaStream_t stream, char* output0)
{
    std::ifstream bin_file("./Constant/Constant_29_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_29_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[2304];
    bin_file.read(tmp_mem, 2304);
    cudaMemcpyAsync(output0, tmp_mem, 2304, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 BlockFusion
// Input:
//	- name: Reshape_12_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Reshape_13_0	type: float	shape: Shape{12, 64, 64}
//	- name: Reshape_9_0	type: float	shape: Shape{12, 64, 64}
//	- name: Reshape_23_0	type: float	shape: Shape{12, 64, 64}
//	- name: Reshape_39_0	type: float	shape: Shape{1}
// Output:
//	- name: BatchMatMul_14_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: BatchMatMul_10_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: BatchMatMul_24_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Broadcast_40_0	type: float	shape: Shape{64, 12, 1, 64}
// Fused functions:
// BatchMatMul_float_float_float_cuda_BatchMatMul_14<<<dim3(192, 1, 1), dim3(128, 1, 1), 0, 0>>>(Reshape_12_0, Reshape_13_0, BatchMatMul_14_0);
// BatchMatMul_float_float_float_cuda_BatchMatMul_10<<<dim3(192, 1, 1), dim3(128, 1, 1), 0, 0>>>(Reshape_12_0, Reshape_9_0, BatchMatMul_10_0);
// BatchMatMul_float_float_float_cuda_BatchMatMul_24<<<dim3(192, 1, 1), dim3(128, 1, 1), 0, 0>>>(Reshape_12_0, Reshape_23_0, BatchMatMul_24_0);
// Broadcast_float_float_cuda_Broadcast_40<<<dim3(384, 1, 1), dim3(128, 1, 1), 0, 0>>>(Reshape_39_0, Broadcast_40_0);
// Deduped function map: <src_function_name : deduped_function_name>
// BatchMatMul_float_float_float_cuda_BatchMatMul_10 : BatchMatMul_float_float_float_cuda_BatchMatMul_14
// BatchMatMul_float_float_float_cuda_BatchMatMul_24 : BatchMatMul_float_float_float_cuda_BatchMatMul_14

// Node name:	BatchMatMul_14
// Description:	BatchMatMul
// Input:
//	- name: Reshape_12_0	type: float	shape: Shape{64, 12, 1, 64}
//	- name: Reshape_13_0	type: float	shape: Shape{12, 64, 64}
// Output:
//	- name: BatchMatMul_14_0	type: float	shape: Shape{64, 12, 1, 64}
__device__ __noinline__ void BatchMatMul_float_float_float_cuda_BatchMatMul_14_block_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        for (int i = 0; i < 1; i++) __syncthreads();
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(192, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    float* A_shared = (float*)(shared_buffer + 0);
    float* B_shared = (float*)(shared_buffer + 4096);
    {
        {
          float compute_local[2];
          
          
          float A_shared_local[1];
          float B_shared_local[2];
          compute_local[(0)] = 0.000000e+00f;
          compute_local[(1)] = 0.000000e+00f;
          A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)))];
          A_shared[((((int)threadIdx.x) + 128))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 1536))];
          A_shared[((((int)threadIdx.x) + 256))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 3072))];
          A_shared[((((int)threadIdx.x) + 384))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 4608))];
          A_shared[((((int)threadIdx.x) + 512))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 6144))];
          A_shared[((((int)threadIdx.x) + 640))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 7680))];
          A_shared[((((int)threadIdx.x) + 768))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 9216))];
          A_shared[((((int)threadIdx.x) + 896))] = A[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 6) * 768)) + (((((int)blockIdx.x) % 48) >> 2) * 64)) + (((int)threadIdx.x) & 63)) + 10752))];
          B_shared[(((int)threadIdx.x))] = B[(((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)))];
          B_shared[((((int)threadIdx.x) + 128))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 512))];
          B_shared[((((int)threadIdx.x) + 256))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1024))];
          B_shared[((((int)threadIdx.x) + 384))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 1536))];
          B_shared[((((int)threadIdx.x) + 512))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2048))];
          B_shared[((((int)threadIdx.x) + 640))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 2560))];
          B_shared[((((int)threadIdx.x) + 768))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3072))];
          B_shared[((((int)threadIdx.x) + 896))] = B[((((((((((int)blockIdx.x) % 48) >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + (((int)threadIdx.x) & 15)) + 3584))];
          __syncthreads();
          for (int k_inner_outer = 0; k_inner_outer < 64; ++k_inner_outer) {
            A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 3) * 64) + k_inner_outer))];
            B_shared_local[(0)] = B_shared[(((k_inner_outer * 16) + (((int)threadIdx.x) & 7)))];
            B_shared_local[(1)] = B_shared[((((k_inner_outer * 16) + (((int)threadIdx.x) & 7)) + 8))];
            compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
            compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
          }
          compute[((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)))] = compute_local[(0)];
          compute[(((((((((int)blockIdx.x) / 48) * 12288) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)blockIdx.x) % 48) * 16)) + (((int)threadIdx.x) & 7)) + 8))] = compute_local[(1)];
        }


    }

}
// Node name:	Broadcast_40
// Description:	Broadcast
// Input:
//	- name: Reshape_39_0	type: float	shape: Shape{1}
// Output:
//	- name: Broadcast_40_0	type: float	shape: Shape{64, 12, 1, 64}
__device__ __noinline__ void Broadcast_float_float_cuda_Broadcast_40_block_kernel(float* input0, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 128){
        return;
    }
    const dim3 blockDim(128, 1, 1);
    const dim3 gridDim(384, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    size_t nthreads = 49152;
    uint32_t strides0 = 768;
    uint32_t strides1 = 64;
    uint32_t strides2 = 64;
    uint32_t strides3 = 1;
    int stride_magic0 = 715827883;
    int stride_magic1 = 1;
    int stride_magic2 = 1;
    int stride_magic3 = 1;
    int stride_shift0 = 7;
    int stride_shift1 = 6;
    int stride_shift2 = 6;
    int stride_shift3 = 0;
    uint32_t reduced_strides0 = 0;
    uint32_t reduced_strides1 = 0;
    uint32_t reduced_strides2 = 1;
    uint32_t reduced_strides3 = 0;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nthreads)
    {
        int coordinate_product = tid;
        int coordinate0 = division_by_invariant_multiplication(coordinate_product, stride_magic0, stride_shift0);
        coordinate_product -= (coordinate0 * strides0);
        int coordinate1 = division_by_invariant_multiplication(coordinate_product, stride_magic1, stride_shift1);
        coordinate_product -= (coordinate1 * strides1);
        int coordinate2 = division_by_invariant_multiplication(coordinate_product, stride_magic2, stride_shift2);
        coordinate_product -= (coordinate2 * strides2);
        int coordinate3 = division_by_invariant_multiplication(coordinate_product, stride_magic3, stride_shift3);
        coordinate_product -= (coordinate3 * strides3);
        uint32_t reduced_idx = 0;
        reduced_idx += coordinate0 * reduced_strides0;
        reduced_idx += coordinate1 * reduced_strides1;
        reduced_idx += coordinate2 * reduced_strides2;
        reduced_idx += coordinate3 * reduced_strides3;
        output0[tid] = load(input0, reduced_idx);
    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_BatchMatMul_Broadcast_1(float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3)
{
    __shared__ char shared_buffer[8192];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 191)
    {
        BatchMatMul_float_float_float_cuda_BatchMatMul_14_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 383)
    {
        BatchMatMul_float_float_float_cuda_BatchMatMul_14_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 192 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 575)
    {
        BatchMatMul_float_float_float_cuda_BatchMatMul_14_block_kernel(input0, input3, output2, threadIdx.x, blockIdx.x - 384 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 576 && (int)blockIdx.x <= 959)
    {
        Broadcast_float_float_cuda_Broadcast_40_block_kernel(input4, output3, threadIdx.x, blockIdx.x - 576 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_BatchMatMul_Broadcast_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* output0, float* output1, float* output2, float* output3) {
    BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_BatchMatMul_Broadcast_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0, output1, output2, output3);
}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 4
#define NNFUSION_GRAPH_OUTPUT_NUM 4
#define NNFUSION_GRAPH_INPUT_DTYPE_0 int64_t
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {}
#define NNFUSION_GRAPH_INPUT_DTYPE_1 float
#define NNFUSION_GRAPH_INPUT_SHAPE_1 {64, 12, 64, 64}
#define NNFUSION_GRAPH_INPUT_DTYPE_2 float
#define NNFUSION_GRAPH_INPUT_SHAPE_2 {64, 12, 1, 64}
#define NNFUSION_GRAPH_INPUT_DTYPE_3 float
#define NNFUSION_GRAPH_INPUT_SHAPE_3 {64, 12, 64, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 int64_t
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_1 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_1 {64, 12, 64, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_2 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_2 {64, 12, 1, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_3 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_3 {64, 12, 64, 64}
#endif

// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN
int get_device_type()
{
    return 0;
}
// Node name:	Constant_18
// Description:	Constant
// Input:
// Output:
//	- name: Constant_18_0	type: int64_t	shape: Shape{64, 12, 3}
void Constant_int64_t_cuda_Constant_18(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_18_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_18_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[18432];
    bin_file.read(tmp_mem, 18432);
    cudaMemcpyAsync(output0, tmp_mem, 18432, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: Constant_38_0	type: float	shape: Shape{}
void Constant_float_cuda_Constant_38(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_38_0 failed.\n");
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
// total memory:2634432
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,1806336));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 1806336));
Reshape_12_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Broadcast_40_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+196608);
BatchMatMul_24_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+393216);
BatchMatMul_10_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+589824);
BatchMatMul_14_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+786432);
Reshape_25_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+393216);
Reshape_27_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+393216);
Select_30_0 = (int64_t*)(group_0_CUDA_GPU0_allocator_memory_pool+983040);
Select_20_0 = (int64_t*)(group_0_CUDA_GPU0_allocator_memory_pool+1001472);
Reshape_15_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+786432);
Reshape_17_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+786432);
Reshape_11_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+589824);
Reshape_32_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+589824);
Reshape_34_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+589824);
BatchMatMul_35_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1019904);
Reshape_36_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1019904);
Reshape_37_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1019904);
Multiply_41_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1216512);
Softmax_42_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1216512);
Reshape_43_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1216512);
BatchMatMul_45_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1413120);
Reshape_46_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1413120);
Reshape_47_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1413120);
BatchMatMul_49_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1609728);
x_2 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1609728);
Result_55_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1609728);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,828096));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 828096));
Constant_38_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Reshape_39_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
Reshape_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
Constant_0_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196672);
Reshape_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+196672);
Constant_1_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+393280);
Reshape_13_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+393280);
Constant_51_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+589888);
Constant_28_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+589952);
Constant_29_0 = (char*)(group_persist_CUDA_GPU0_allocator_memory_pool+608384);
Constant_18_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+610688);
Constant_19_0 = (char*)(group_persist_CUDA_GPU0_allocator_memory_pool+629120);
gen_id_1 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+631424);
// tensor_31 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
// tensor_21 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
// Result_56_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+631488);
Reshape_48_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+631488);
// Reshape_44_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
// Reshape_33_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
// Result_54_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
Result_53_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+631424);
// create streams/handles
 // name=@tmp_14
Constant_float_cuda_Constant_38(0, Constant_38_0);
 // name=weight_v_0
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=weight_q_0
Constant_float_cuda_Constant_0(0, Constant_0_0);
 // name=weight_k_0
Constant_float_cuda_Constant_1(0, Constant_1_0);
 // name=@tmp_15
Constant_int64_t_cuda_Constant_51(0, Constant_51_0);
 // name=@tmp_10
Constant_int64_t_cuda_Constant_28(0, Constant_28_0);
 // name=@tmp_11
Constant_char_cuda_Constant_29(0, Constant_29_0);
 // name=@tmp_3
Constant_int64_t_cuda_Constant_18(0, Constant_18_0);
 // name=@tmp_4
Constant_char_cuda_Constant_19(0, Constant_19_0);
 // name=weight_o_0
Constant_float_cuda_Constant_3(0, Constant_3_0);
}


extern "C" int kernel_entry(int64_t* Parameter_4_0, float* Parameter_5_0, float* Parameter_6_0, float* Parameter_7_0, int64_t* Result_53_0, float* Result_54_0, float* Result_55_0, float* Result_56_0)
{
// kernel_entry_init
 // name=Reshape_39
// eliminated: Reshape_float_float_cuda_Reshape_39_Call(dim3(1, 1, 1), dim3(128, 1, 1), 0, 0, Constant_38_0, Reshape_39_0);
 // name=Reshape_23
// eliminated: Reshape_float_float_cuda_Reshape_23_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Constant_2_0, Reshape_23_0);
 // name=Reshape_12
Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Parameter_6_0, Reshape_12_0);
 // name=Reshape_9
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Constant_0_0, Reshape_9_0);
 // name=Reshape_13
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Constant_1_0, Reshape_13_0);
 // name=blockfusion_kernel_58
BlockFusionKernel_float_float_float_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_BatchMatMul_Broadcast_1_Call(dim3(960, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_12_0, Reshape_13_0, Reshape_9_0, Reshape_23_0, Reshape_39_0, BatchMatMul_14_0, BatchMatMul_10_0, BatchMatMul_24_0, Broadcast_40_0);
 // name=Reshape_25
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_24_0, Reshape_25_0);
 // name=@tmp_9
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_25_0, Reshape_27_0);
 // name=blockfusion_kernel_57
BlockFusionKernel_char_int64_t_int64_t_char_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_Select_Select_Add_0_Call(dim3(37, 1, 1), dim3(128, 1, 1), 0, 0, Constant_19_0, Constant_18_0, Parameter_4_0, Constant_29_0, Constant_28_0, Constant_51_0, Select_20_0, Select_30_0, Result_53_0);
 // name=Reshape_15
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_14_0, Reshape_15_0);
 // name=@tmp_2
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_15_0, Reshape_17_0);
 // name=blockfusion_kernel_59
tensor_21 = Parameter_7_0;
tensor_31 = Parameter_5_0;
/* memref */BlockFusionKernel_float_int64_t_float_float_int64_t_float_float_float_cuda_ScatterND_ScatterND_2_Call(dim3(768, 1, 1), dim3(128, 1, 1), 0, 0, Parameter_7_0, Select_20_0, Reshape_17_0, Parameter_5_0, Select_30_0, Reshape_27_0, Result_56_0, Result_54_0);
 // name=Result_56
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_56(0, Result_56_0, Result_56_0);
 // name=Reshape_48
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Constant_3_0, Reshape_48_0);
 // name=Reshape_44
// Reshape_float_float_cuda_Reshape_44_Call(dim3(24576, 1, 1), dim3(128, 1, 1), 0, 0, Result_54_0, Reshape_44_0);
 // name=Reshape_11
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_10_0, Reshape_11_0);
 // name=matmul_arg1_3
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_11_0, Reshape_32_0);
 // name=Reshape_34
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_32_0, Reshape_34_0);
 // name=Reshape_33
// Reshape_float_float_cuda_Reshape_44_Call(dim3(24576, 1, 1), dim3(128, 1, 1), 0, 0, Result_56_0, Reshape_33_0);
 // name=sum_arg00_0
BatchMatMul_float_float_float_cuda_BatchMatMul_35_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Parameter_7_0, Reshape_34_0, BatchMatMul_35_0);
 // name=Reshape_36
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_35_0, Reshape_36_0);
 // name=attn_0
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_36_0, Reshape_37_0);
 // name=attn_1
Multiply_float_float_float_cuda_Multiply_41_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_37_0, Broadcast_40_0, Multiply_41_0);
 // name=attn_2
Softmax_float_float_cuda_Softmax_42_Call(dim3(192, 1, 1), dim3(128, 1, 1), 0, 0, Multiply_41_0, Softmax_42_0);
 // name=Reshape_43
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Softmax_42_0, Reshape_43_0);
 // name=x_1
BatchMatMul_float_float_float_cuda_BatchMatMul_45_Call(dim3(384, 1, 1), dim3(64, 1, 1), 0, 0, Reshape_43_0, Parameter_5_0, BatchMatMul_45_0);
 // name=Reshape_46
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_45_0, Reshape_46_0);
 // name=Reshape_47
// eliminated: Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_46_0, Reshape_47_0);
 // name=x_2
BatchMatMul_float_float_float_cuda_BatchMatMul_49_Call(dim3(192, 1, 1), dim3(128, 1, 1), 0, 0, Reshape_47_0, Reshape_48_0, BatchMatMul_49_0);
 // name=Reshape_50
Reshape_float_float_cuda_Reshape_12_Call(dim3(384, 1, 1), dim3(128, 1, 1), 0, 0, BatchMatMul_49_0, Result_55_0);
 // name=Result_55
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_55(0, Result_55_0, Result_55_0);
 // name=Result_54
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_56(0, Result_54_0, Result_54_0);
 // name=Result_53
// eliminated (extern_result_memory): Result_int64_t_int64_t_cuda_lib_Result_53(0, Result_53_0, Result_53_0);
return 0;
}


extern "C" void cuda_free()
{
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_0_CUDA_GPU0_allocator_memory_pool));
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaFree(group_persist_CUDA_GPU0_allocator_memory_pool));
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

