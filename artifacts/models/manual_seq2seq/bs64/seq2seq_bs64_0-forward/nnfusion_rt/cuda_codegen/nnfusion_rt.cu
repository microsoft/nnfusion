#include <stdexcept>
#include <assert.h>
#include <cublas_v2.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <sstream>
#include "nnfusion_rt.h"
#include <fstream>

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
#define MIN(a,b) ((a)>(b)?(b):(a))
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
__device__ __forceinline__ char logical_and(char x0, char x1)
{
    return x0 & x1;
}
__device__ __forceinline__ char greater(int64_t x0, int64_t x1)
{
    return x0 > x1;
}

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, delta, width);
#endif
}

__device__ static float reduceMax(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val = max(val, CudaShuffleDownSync(mask, val, 16));
  val = max(val, CudaShuffleDownSync(mask, val, 8));
  val = max(val, CudaShuffleDownSync(mask, val, 4));
  val = max(val, CudaShuffleDownSync(mask, val, 2));
  val = max(val, CudaShuffleDownSync(mask, val, 1));

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));
  }

  return val;
}

__device__ static float reduceSum(float val, int tid, int blockSize, float* shm) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < blockSize);

  val += CudaShuffleDownSync(mask, val, 16);
  val += CudaShuffleDownSync(mask, val, 8);
  val += CudaShuffleDownSync(mask, val, 4);
  val += CudaShuffleDownSync(mask, val, 2);
  val += CudaShuffleDownSync(mask, val, 1);

  if (tid < warpSize) shm[tid] = 0.;
  __syncthreads();

  if (tid % warpSize == 0) shm[tid / warpSize] = val;

  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);
  }

  return val;
}
__device__ __forceinline__ float sigmoid(float x0)
{
    return 1 / (1 + expf(-x0));
}
__device__ __forceinline__ int64_t add(int64_t x0, int64_t x1)
{
    return x0 + x1;
}
__device__ __forceinline__ char nnfusion_less(int64_t x0, int64_t x1)
{
    return x0 < x1;
}
__device__ __forceinline__ float mul(float x0, float x1)
{
    return x0 * x1;
}
__device__ __forceinline__ float add(float x0, float x1)
{
    return x0 + x1;
}

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
int num_SMs;
char* group_0_CUDA_GPU0_allocator_memory_pool;
float* GatherV2_19_0;
float* GatherV2_82_0;
float* Reshape_22_0;
float* Reshape_21_0;
float* Reshape_26_0;
float* BatchMatMul_24_0;
float* BatchMatMul_28_0;
float* Reshape_25_0;
float* Reshape_29_0;
float* GatherV2_55_0;
float* GatherV2_47_0;
float* GatherV2_39_0;
float* GatherV2_31_0;
float* GatherV2_59_0;
float* GatherV2_51_0;
float* GatherV2_43_0;
float* GatherV2_35_0;
float* Dot_77_0;
float* Add_83_0;
char* Less_94_0;
int64_t* Reshape_86_0;
int64_t* Sum_90_0;
char* Greater_92_0;
char* group_persist_CUDA_GPU0_allocator_memory_pool;
int64_t* Constant_88_0;
float* Constant_0_0;
int64_t* id_1;
float* Constant_11_0;
int64_t* Constant_54_0;
float* Constant_1_0;
float* Reshape_23_0;
float* Constant_2_0;
float* Reshape_27_0;
int64_t* Constant_46_0;
int64_t* Constant_38_0;
int64_t* Constant_30_0;
int64_t* Constant_58_0;
int64_t* Constant_50_0;
int64_t* Constant_42_0;
int64_t* Constant_34_0;
float* Constant_9_0;
float* Constant_10_0;
float* Constant_5_0;
float* Constant_6_0;
float* Constant_3_0;
float* Constant_4_0;
float* Constant_7_0;
float* Constant_8_0;
float* h_2;
float* c_0;
float* Reshape_76_0;
float* Constant_12_0;
int64_t* tensor_84;
int64_t* Result_101_0;
int64_t* Result_100_0;
int64_t* Constant_93_0;
int64_t* Constant_91_0;
int64_t* tensor_87;
char* cond_0;
char* Result_99_0;
int64_t* Result_98_0;
float* Result_97_0;
float* Result_96_0;
// Node name:	Constant_50
// Description:	Constant
// Input:
// Output:
//	- name: Constant_50_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_50(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_50_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_50_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_5
// Description:	Constant
// Input:
// Output:
//	- name: Constant_5_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_5(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_5_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_5_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Less_94
// Description:	Less
// Input:
//	- name: id_1	type: int64_t	shape: Shape{}
//	- name: Constant_93_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Less_94_0	type: char	shape: Shape{}
extern "C" __launch_bounds__(1) __global__ void Less_int64_t_int64_t_char_cuda_Less_94(int64_t* input0, int64_t* input1, char* output0)
{
    output0[threadIdx.x] = nnfusion_less(input0[0], input1[0]);

}
extern void Less_int64_t_int64_t_char_cuda_Less_94_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int64_t* input0, int64_t* input1, char* output0) {
    Less_int64_t_int64_t_char_cuda_Less_94<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Result_97
// Description:	Result
// Input:
//	- name: c_0	type: float	shape: Shape{64, 256}
// Output:
//	- name: Result_97_0	type: float	shape: Shape{64, 256}
void Result_float_float_cuda_lib_Result_97(cudaStream_t stream, float* input0, float* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,65536, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_8_0	type: float	shape: Shape{256}
//	- name: GatherV2_51_0	type: float	shape: Shape{64, 256}
//	- name: Constant_7_0	type: float	shape: Shape{256}
//	- name: GatherV2_47_0	type: float	shape: Shape{64, 256}
//	- name: Constant_4_0	type: float	shape: Shape{256}
//	- name: GatherV2_35_0	type: float	shape: Shape{64, 256}
//	- name: Constant_3_0	type: float	shape: Shape{256}
//	- name: GatherV2_31_0	type: float	shape: Shape{64, 256}
//	- name: Constant_6_0	type: float	shape: Shape{256}
//	- name: GatherV2_43_0	type: float	shape: Shape{64, 256}
//	- name: Constant_5_0	type: float	shape: Shape{256}
//	- name: GatherV2_39_0	type: float	shape: Shape{64, 256}
//	- name: Parameter_14_0	type: float	shape: Shape{64, 256}
//	- name: Constant_10_0	type: float	shape: Shape{256}
//	- name: GatherV2_59_0	type: float	shape: Shape{64, 256}
//	- name: Constant_9_0	type: float	shape: Shape{256}
//	- name: GatherV2_55_0	type: float	shape: Shape{64, 256}
// Output:
//	- name: c_0	type: float	shape: Shape{64, 256}
//	- name: h_2	type: float	shape: Shape{64, 256}
// Fused functions:
// Broadcast, Broadcast_52
// Add, hh2_0
// Broadcast, Broadcast_48
// Add, ih2_0
// Add, Tanh_arg00_0
// Tanh, cellgate_0
// Broadcast, Broadcast_36
// Add, hh0_0
// Broadcast, Broadcast_32
// Add, ih0_0
// Add, Sigmoid_arg00_0
// Sigmoid, ingate_0
// Multiply, @tmp_29
// Broadcast, Broadcast_44
// Add, hh1_0
// Broadcast, Broadcast_40
// Add, ih1_0
// Add, Sigmoid_arg00_1
// Sigmoid, forgetgate_0
// Multiply, @tmp_28
// Add, c_0
// Tanh, @tmp_30
// Broadcast, Broadcast_60
// Add, hh3_0
// Broadcast, Broadcast_56
// Add, ih3_0
// Add, Sigmoid_arg00_2
// Sigmoid, outgate_0
// Multiply, h_2
extern "C" __launch_bounds__(256) __global__ void FusedKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Broadcast_Add_Broadcast_Add_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_0(float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1)
{
    int tid = blockIdx.x * 256 + threadIdx.x;
    float temp0 = input0[tid % 256];
    float temp1 = add(input1[tid], temp0);
    float temp2 = input2[tid % 256];
    float temp3 = add(input3[tid], temp2);
    float temp4 = add(temp3, temp1);
    float temp5 = tanhf(temp4);
    float temp6 = input4[tid % 256];
    float temp7 = add(input5[tid], temp6);
    float temp8 = input6[tid % 256];
    float temp9 = add(input7[tid], temp8);
    float temp10 = add(temp9, temp7);
    float temp11 = sigmoid(temp10);
    float temp12 = mul(temp11, temp5);
    float temp13 = input8[tid % 256];
    float temp14 = add(input9[tid], temp13);
    float temp15 = input10[tid % 256];
    float temp16 = add(input11[tid], temp15);
    float temp17 = add(temp16, temp14);
    float temp18 = sigmoid(temp17);
    float temp19 = mul(temp18, input12[tid]);
    float temp20 = add(temp19, temp12);
    float temp21 = tanhf(temp20);
    float temp22 = input13[tid % 256];
    float temp23 = add(input14[tid], temp22);
    float temp24 = input15[tid % 256];
    float temp25 = add(input16[tid], temp24);
    float temp26 = add(temp25, temp23);
    float temp27 = sigmoid(temp26);
    float temp28 = mul(temp27, temp21);
    output1[tid] = temp28;
    output0[tid] = temp20;

}
extern void FusedKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Broadcast_Add_Broadcast_Add_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* input4, float* input5, float* input6, float* input7, float* input8, float* input9, float* input10, float* input11, float* input12, float* input13, float* input14, float* input15, float* input16, float* output0, float* output1) {
    FusedKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Broadcast_Add_Broadcast_Add_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, output0, output1);
}
// Node name:	 BlockFusion
// Input:
//	- name: Reshape_29_0	type: float	shape: Shape{4, 64, 256}
//	- name: Constant_34_0	type: int64_t	shape: Shape{}
//	- name: Constant_42_0	type: int64_t	shape: Shape{}
//	- name: Constant_50_0	type: int64_t	shape: Shape{}
//	- name: Constant_58_0	type: int64_t	shape: Shape{}
//	- name: Reshape_25_0	type: float	shape: Shape{4, 64, 256}
//	- name: Constant_30_0	type: int64_t	shape: Shape{}
//	- name: Constant_38_0	type: int64_t	shape: Shape{}
//	- name: Constant_46_0	type: int64_t	shape: Shape{}
//	- name: Constant_54_0	type: int64_t	shape: Shape{}
// Output:
//	- name: GatherV2_35_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_43_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_51_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_59_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_31_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_39_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_47_0	type: float	shape: Shape{64, 256}
//	- name: GatherV2_55_0	type: float	shape: Shape{64, 256}
// Fused functions:
// GatherV2_float_int64_t_float_cuda_GatherV2_35<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_29_0, Constant_34_0, GatherV2_35_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_43<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_29_0, Constant_42_0, GatherV2_43_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_51<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_29_0, Constant_50_0, GatherV2_51_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_59<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_29_0, Constant_58_0, GatherV2_59_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_31<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_25_0, Constant_30_0, GatherV2_31_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_39<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_25_0, Constant_38_0, GatherV2_39_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_47<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_25_0, Constant_46_0, GatherV2_47_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_55<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_25_0, Constant_54_0, GatherV2_55_0);
// Deduped function map: <src_function_name : deduped_function_name>
// GatherV2_float_int64_t_float_cuda_GatherV2_43 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_51 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_59 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_31 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_39 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_47 : GatherV2_float_int64_t_float_cuda_GatherV2_35
// GatherV2_float_int64_t_float_cuda_GatherV2_55 : GatherV2_float_int64_t_float_cuda_GatherV2_35

// Node name:	GatherV2_35
// Description:	GatherV2
// Input:
//	- name: Reshape_29_0	type: float	shape: Shape{4, 64, 256}
//	- name: Constant_34_0	type: int64_t	shape: Shape{}
// Output:
//	- name: GatherV2_35_0	type: float	shape: Shape{64, 256}
__device__ __noinline__ void GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(float* input0, int64_t* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(256, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    float* params = input0;
    int64_t* indices = input1;
    float* out = output0;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 16384)
    {
        uint32_t batch_i = 0;
        uint32_t indices_i = 0;
        uint32_t slice_i = 0;
        indices_i = i / 16384;
        slice_i = i - indices_i * 16384;
        uint32_t gather_i = *(indices + indices_i);
        if (gather_i >= 4)
           out[i] = 0;
        else
        {
            uint32_t params_i = (batch_i * 4 + gather_i) * 16384 + slice_i;
            out[i] = __ldg(params + params_i);
        }
    }

}
extern "C" __global__  void BlockFusionKernel_float_int64_t_int64_t_int64_t_int64_t_float_int64_t_int64_t_int64_t_int64_t_float_float_float_float_float_float_float_float_cuda_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_2(float* input0, int64_t* input1, int64_t* input2, int64_t* input3, int64_t* input4, float* input5, int64_t* input6, int64_t* input7, int64_t* input8, int64_t* input9, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 63)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 64 && (int)blockIdx.x <= 127)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input0, input2, output1, threadIdx.x, blockIdx.x - 64 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 128 && (int)blockIdx.x <= 191)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input0, input3, output2, threadIdx.x, blockIdx.x - 128 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 192 && (int)blockIdx.x <= 255)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input0, input4, output3, threadIdx.x, blockIdx.x - 192 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 256 && (int)blockIdx.x <= 319)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input5, input6, output4, threadIdx.x, blockIdx.x - 256 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 320 && (int)blockIdx.x <= 383)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input5, input7, output5, threadIdx.x, blockIdx.x - 320 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 384 && (int)blockIdx.x <= 447)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input5, input8, output6, threadIdx.x, blockIdx.x - 384 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 448 && (int)blockIdx.x <= 511)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_35_block_kernel(input5, input9, output7, threadIdx.x, blockIdx.x - 448 + 0, NULL);
    }

}
extern void BlockFusionKernel_float_int64_t_int64_t_int64_t_int64_t_float_int64_t_int64_t_int64_t_int64_t_float_float_float_float_float_float_float_float_cuda_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_2_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int64_t* input1, int64_t* input2, int64_t* input3, int64_t* input4, float* input5, int64_t* input6, int64_t* input7, int64_t* input8, int64_t* input9, float* output0, float* output1, float* output2, float* output3, float* output4, float* output5, float* output6, float* output7) {
    BlockFusionKernel_float_int64_t_int64_t_int64_t_int64_t_float_int64_t_int64_t_int64_t_int64_t_float_float_float_float_float_float_float_float_cuda_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_2<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, input5, input6, input7, input8, input9, output0, output1, output2, output3, output4, output5, output6, output7);
}
// Node name:	And_95
// Description:	And
// Input:
//	- name: Greater_92_0	type: char	shape: Shape{}
//	- name: Less_94_0	type: char	shape: Shape{}
// Output:
//	- name: cond_0	type: char	shape: Shape{}
extern "C" __launch_bounds__(1) __global__ void And_char_char_char_cuda_And_95(char* input0, char* input1, char* output0)
{
    output0[threadIdx.x] = logical_and(input0[0], input1[0]);

}
extern void And_char_char_char_cuda_And_95_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, char* input0, char* input1, char* output0) {
    And_char_char_char_cuda_And_95<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Result_101
// Description:	Result
// Input:
//	- name: tensor_84	type: int64_t	shape: Shape{64}
// Output:
//	- name: Result_101_0	type: int64_t	shape: Shape{64}
void Result_int64_t_int64_t_cuda_lib_Result_101(cudaStream_t stream, int64_t* input0, int64_t* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,512, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	 BlockFusion
// Input:
//	- name: Reshape_26_0	type: float	shape: Shape{64, 256}
//	- name: Reshape_27_0	type: float	shape: Shape{4, 256, 256}
//	- name: Reshape_22_0	type: float	shape: Shape{64, 256}
//	- name: Reshape_23_0	type: float	shape: Shape{4, 256, 256}
// Output:
//	- name: BatchMatMul_28_0	type: float	shape: Shape{4, 64, 256}
//	- name: BatchMatMul_24_0	type: float	shape: Shape{4, 64, 256}
// Fused functions:
// BatchMatMul_float_float_float_cuda_BatchMatMul_28<<<dim3(8, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_26_0, Reshape_27_0, BatchMatMul_28_0);
// BatchMatMul_float_float_float_cuda_BatchMatMul_24<<<dim3(8, 1, 1), dim3(256, 1, 1), 0, 0>>>(Reshape_22_0, Reshape_23_0, BatchMatMul_24_0);
// Deduped function map: <src_function_name : deduped_function_name>
// BatchMatMul_float_float_float_cuda_BatchMatMul_24 : BatchMatMul_float_float_float_cuda_BatchMatMul_28

// Node name:	BatchMatMul_28
// Description:	BatchMatMul
// Input:
//	- name: Reshape_26_0	type: float	shape: Shape{64, 256}
//	- name: Reshape_27_0	type: float	shape: Shape{4, 256, 256}
// Output:
//	- name: BatchMatMul_28_0	type: float	shape: Shape{4, 64, 256}
__device__ __noinline__ void BatchMatMul_float_float_float_cuda_BatchMatMul_28_block_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        for (int i = 0; i < 16; i++) __syncthreads();
        return;
    }
    const dim3 blockDim(256, 1, 1);
    const dim3 gridDim(8, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    float* A_shared = (float*)(shared_buffer + 0);
    float* B_shared = (float*)(shared_buffer + 8192);
    {
        {
          float compute_local[32];
          
          
          float A_shared_local[4];
          float B_shared_local[8];
          compute_local[(0)] = 0.000000e+00f;
          compute_local[(8)] = 0.000000e+00f;
          compute_local[(16)] = 0.000000e+00f;
          compute_local[(24)] = 0.000000e+00f;
          compute_local[(2)] = 0.000000e+00f;
          compute_local[(10)] = 0.000000e+00f;
          compute_local[(18)] = 0.000000e+00f;
          compute_local[(26)] = 0.000000e+00f;
          compute_local[(4)] = 0.000000e+00f;
          compute_local[(12)] = 0.000000e+00f;
          compute_local[(20)] = 0.000000e+00f;
          compute_local[(28)] = 0.000000e+00f;
          compute_local[(6)] = 0.000000e+00f;
          compute_local[(14)] = 0.000000e+00f;
          compute_local[(22)] = 0.000000e+00f;
          compute_local[(30)] = 0.000000e+00f;
          compute_local[(1)] = 0.000000e+00f;
          compute_local[(9)] = 0.000000e+00f;
          compute_local[(17)] = 0.000000e+00f;
          compute_local[(25)] = 0.000000e+00f;
          compute_local[(3)] = 0.000000e+00f;
          compute_local[(11)] = 0.000000e+00f;
          compute_local[(19)] = 0.000000e+00f;
          compute_local[(27)] = 0.000000e+00f;
          compute_local[(5)] = 0.000000e+00f;
          compute_local[(13)] = 0.000000e+00f;
          compute_local[(21)] = 0.000000e+00f;
          compute_local[(29)] = 0.000000e+00f;
          compute_local[(7)] = 0.000000e+00f;
          compute_local[(15)] = 0.000000e+00f;
          compute_local[(23)] = 0.000000e+00f;
          compute_local[(31)] = 0.000000e+00f;
          for (int k_outer = 0; k_outer < 8; ++k_outer) {
            __syncthreads();
            A_shared[(((int)threadIdx.x))] = A[(((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)))];
            A_shared[((((int)threadIdx.x) + 256))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
            A_shared[((((int)threadIdx.x) + 512))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
            A_shared[((((int)threadIdx.x) + 768))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
            A_shared[((((int)threadIdx.x) + 1024))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192))];
            A_shared[((((int)threadIdx.x) + 1280))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240))];
            A_shared[((((int)threadIdx.x) + 1536))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288))];
            A_shared[((((int)threadIdx.x) + 1792))] = A[((((((((int)threadIdx.x) >> 5) * 256) + (k_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336))];
            B_shared[(((int)threadIdx.x))] = B[(((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 256))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 2048))];
            B_shared[((((int)threadIdx.x) + 512))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 4096))];
            B_shared[((((int)threadIdx.x) + 768))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 6144))];
            B_shared[((((int)threadIdx.x) + 1024))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 65536))];
            B_shared[((((int)threadIdx.x) + 1280))] = B[((((((((((int)threadIdx.x) + 1280) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 1536))] = B[((((((((((int)threadIdx.x) + 1536) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 1792))] = B[((((((((((int)threadIdx.x) + 1792) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 2048))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 131072))];
            B_shared[((((int)threadIdx.x) + 2304))] = B[((((((((((int)threadIdx.x) + 2304) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 2560))] = B[((((((((((int)threadIdx.x) + 2560) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 2816))] = B[((((((((((int)threadIdx.x) + 2816) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 3072))] = B[((((((k_outer * 8192) + ((((int)threadIdx.x) >> 5) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)) + 196608))];
            B_shared[((((int)threadIdx.x) + 3328))] = B[((((((((((int)threadIdx.x) + 3328) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 8) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 3584))] = B[((((((((((int)threadIdx.x) + 3584) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 16) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            B_shared[((((int)threadIdx.x) + 3840))] = B[((((((((((int)threadIdx.x) + 3840) >> 10) * 65536) + (k_outer * 8192)) + (((((int)threadIdx.x) >> 5) + 24) * 256)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 31)))];
            __syncthreads();
            for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
              A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer))];
              A_shared_local[(1)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 512))];
              A_shared_local[(2)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 1024))];
              A_shared_local[(3)] = A_shared[(((((((int)threadIdx.x) >> 4) * 32) + k_inner_outer) + 1536))];
              B_shared_local[(0)] = B_shared[(((k_inner_outer * 32) + (((int)threadIdx.x) & 15)))];
              B_shared_local[(2)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 1024))];
              B_shared_local[(4)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 2048))];
              B_shared_local[(6)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 3072))];
              B_shared_local[(1)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 16))];
              B_shared_local[(3)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 1040))];
              B_shared_local[(5)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 2064))];
              B_shared_local[(7)] = B_shared[((((k_inner_outer * 32) + (((int)threadIdx.x) & 15)) + 3088))];
              compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
              compute_local[(8)] = (compute_local[(8)] + (A_shared_local[(0)] * B_shared_local[(2)]));
              compute_local[(16)] = (compute_local[(16)] + (A_shared_local[(0)] * B_shared_local[(4)]));
              compute_local[(24)] = (compute_local[(24)] + (A_shared_local[(0)] * B_shared_local[(6)]));
              compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
              compute_local[(10)] = (compute_local[(10)] + (A_shared_local[(1)] * B_shared_local[(2)]));
              compute_local[(18)] = (compute_local[(18)] + (A_shared_local[(1)] * B_shared_local[(4)]));
              compute_local[(26)] = (compute_local[(26)] + (A_shared_local[(1)] * B_shared_local[(6)]));
              compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
              compute_local[(12)] = (compute_local[(12)] + (A_shared_local[(2)] * B_shared_local[(2)]));
              compute_local[(20)] = (compute_local[(20)] + (A_shared_local[(2)] * B_shared_local[(4)]));
              compute_local[(28)] = (compute_local[(28)] + (A_shared_local[(2)] * B_shared_local[(6)]));
              compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
              compute_local[(14)] = (compute_local[(14)] + (A_shared_local[(3)] * B_shared_local[(2)]));
              compute_local[(22)] = (compute_local[(22)] + (A_shared_local[(3)] * B_shared_local[(4)]));
              compute_local[(30)] = (compute_local[(30)] + (A_shared_local[(3)] * B_shared_local[(6)]));
              compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
              compute_local[(9)] = (compute_local[(9)] + (A_shared_local[(0)] * B_shared_local[(3)]));
              compute_local[(17)] = (compute_local[(17)] + (A_shared_local[(0)] * B_shared_local[(5)]));
              compute_local[(25)] = (compute_local[(25)] + (A_shared_local[(0)] * B_shared_local[(7)]));
              compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
              compute_local[(11)] = (compute_local[(11)] + (A_shared_local[(1)] * B_shared_local[(3)]));
              compute_local[(19)] = (compute_local[(19)] + (A_shared_local[(1)] * B_shared_local[(5)]));
              compute_local[(27)] = (compute_local[(27)] + (A_shared_local[(1)] * B_shared_local[(7)]));
              compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
              compute_local[(13)] = (compute_local[(13)] + (A_shared_local[(2)] * B_shared_local[(3)]));
              compute_local[(21)] = (compute_local[(21)] + (A_shared_local[(2)] * B_shared_local[(5)]));
              compute_local[(29)] = (compute_local[(29)] + (A_shared_local[(2)] * B_shared_local[(7)]));
              compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
              compute_local[(15)] = (compute_local[(15)] + (A_shared_local[(3)] * B_shared_local[(3)]));
              compute_local[(23)] = (compute_local[(23)] + (A_shared_local[(3)] * B_shared_local[(5)]));
              compute_local[(31)] = (compute_local[(31)] + (A_shared_local[(3)] * B_shared_local[(7)]));
            }
          }
          compute[(((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)))] = compute_local[(0)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16384))] = compute_local[(8)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 32768))] = compute_local[(16)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 49152))] = compute_local[(24)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 4096))] = compute_local[(2)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 20480))] = compute_local[(10)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 36864))] = compute_local[(18)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 53248))] = compute_local[(26)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 8192))] = compute_local[(4)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 24576))] = compute_local[(12)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 40960))] = compute_local[(20)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 57344))] = compute_local[(28)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 12288))] = compute_local[(6)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 28672))] = compute_local[(14)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 45056))] = compute_local[(22)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 61440))] = compute_local[(30)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16))] = compute_local[(1)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 16400))] = compute_local[(9)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 32784))] = compute_local[(17)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 49168))] = compute_local[(25)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 4112))] = compute_local[(3)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 20496))] = compute_local[(11)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 36880))] = compute_local[(19)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 53264))] = compute_local[(27)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 8208))] = compute_local[(5)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 24592))] = compute_local[(13)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 40976))] = compute_local[(21)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 57360))] = compute_local[(29)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 12304))] = compute_local[(7)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 28688))] = compute_local[(15)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 45072))] = compute_local[(23)];
          compute[((((((((int)threadIdx.x) >> 4) * 256) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) & 15)) + 61456))] = compute_local[(31)];
        }


    }

}
extern "C" __global__  void BlockFusionKernel_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_1(float* input0, float* input1, float* input2, float* input3, float* output0, float* output1)
{
    __shared__ char shared_buffer[24576];

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 7)
    {
        BatchMatMul_float_float_float_cuda_BatchMatMul_28_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, shared_buffer);
    }
    else if ((int)blockIdx.x >= 8 && (int)blockIdx.x <= 15)
    {
        BatchMatMul_float_float_float_cuda_BatchMatMul_28_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 8 + 0, shared_buffer);
    }

}
extern void BlockFusionKernel_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* input3, float* output0, float* output1) {
    BlockFusionKernel_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, output0, output1);
}
// Node name:	Reshape_21
// Description:	Reshape
// Input:
//	- name: Parameter_13_0	type: float	shape: Shape{64, 256}
// Output:
//	- name: Reshape_21_0	type: float	shape: Shape{64, 256}
extern "C" __launch_bounds__(256) __global__ void Reshape_float_float_cuda_Reshape_21(float* input0, float* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 16384) { return; }
    output0[tid] = input0[tid];

}
extern void Reshape_float_float_cuda_Reshape_21_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* output0) {
    Reshape_float_float_cuda_Reshape_21<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Result_99
// Description:	Result
// Input:
//	- name: cond_0	type: char	shape: Shape{}
// Output:
//	- name: Result_99_0	type: char	shape: Shape{}
void Result_char_char_cuda_lib_Result_99(cudaStream_t stream, char* input0, char* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,1, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	 BlockFusion
// Input:
//	- name: Parameter_16_0	type: int64_t	shape: Shape{50, 64}
//	- name: Reshape_86_0	type: int64_t	shape: Shape{1}
//	- name: tensor_84	type: int64_t	shape: Shape{64}
// Output:
//	- name: tensor_87	type: int64_t	shape: Shape{50, 64}
//	- name: Sum_90_0	type: int64_t	shape: Shape{}
// Fused functions:
// ScatterND_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_87<<<dim3(1, 1, 1), dim3(256, 1, 1), 0, 0>>>(Parameter_16_0, Reshape_86_0, tensor_84, tensor_87);
// Sum_int64_t_int64_t_cuda_Sum_90<<<dim3(1, 1, 1), dim3(64, 1, 1), 0, 0>>>(tensor_84, Sum_90_0);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	ScatterND_87
// Description:	ScatterND
// Input:
//	- name: Parameter_16_0	type: int64_t	shape: Shape{50, 64}
//	- name: Reshape_86_0	type: int64_t	shape: Shape{1}
//	- name: tensor_84	type: int64_t	shape: Shape{64}
// Output:
//	- name: tensor_87	type: int64_t	shape: Shape{50, 64}
__device__ __noinline__ void ScatterND_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_87_block_kernel(int64_t* input0, int64_t* input1, int64_t* input2, int64_t* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(256, 1, 1);
    const dim3 gridDim(1, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64) { return; }
    input0[64* input1[0] + tid] = input2[tid];

}
// Node name:	Sum_90
// Description:	Sum
// Input:
//	- name: tensor_84	type: int64_t	shape: Shape{64}
// Output:
//	- name: Sum_90_0	type: int64_t	shape: Shape{}
__device__ __noinline__ void Sum_int64_t_int64_t_cuda_Sum_90_block_kernel(int64_t* input0, int64_t* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 64){
        for (int i = 0; i < 2; i++) __syncthreads();
        return;
    }
    const dim3 blockDim(64, 1, 1);
    const dim3 gridDim(1, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);

    int width = 64;
    int block_size = 64;
    const int warp_size = 32;
                        float* shm = (float*)(shared_buffer + 128);

    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int data_idx_offset = block_idx * width;

    float val = 0.0;
    for (int tidx = thread_idx; tidx < width; tidx += block_size) {
        int data_idx = tidx + data_idx_offset;
        val += input0[data_idx];
    }
    val = reduceSum(val, thread_idx, block_size, shm);
    if (thread_idx == 0) output0[block_idx] = val;


}
extern "C" __global__  void BlockFusionKernel_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_Sum_3(int64_t* input0, int64_t* input1, int64_t* input2, int64_t* output0, int64_t* output1)
{
    __shared__ char shared_buffer[256];

    if (blockIdx.x == 0)
    {
        ScatterND_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_87_block_kernel(input0, input1, input2, output0, threadIdx.x, 0, shared_buffer);
    }
    else if (blockIdx.x == 1)
    {
        Sum_int64_t_int64_t_cuda_Sum_90_block_kernel(input2, output1, threadIdx.x, 0, shared_buffer);
    }

}
extern void BlockFusionKernel_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_Sum_3_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int64_t* input0, int64_t* input1, int64_t* input2, int64_t* output0, int64_t* output1) {
    BlockFusionKernel_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_Sum_3<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0, output1);
}
// Node name:	ArgMax_84
// Description:	ArgMax
// Input:
//	- name: Add_83_0	type: float	shape: Shape{64, 3797}
// Output:
//	- name: tensor_84	type: int64_t	shape: Shape{64}
extern "C" __launch_bounds__(256) __global__ void ArgMax_float_int64_t_cuda_ArgMax_84(float* input0, int64_t* output0)
{

    int in_reduce_size = 1;
    int reduce_size = 3797;
    int out_id = blockIdx.x / in_reduce_size;
    int in_id = blockIdx.x % in_reduce_size;
    int bias = out_id * reduce_size * in_reduce_size + in_id;
    int max_id = -1;
    float max_value = -FLT_MAX;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        float value = input0[bias + i * in_reduce_size];
        if (value > max_value) {
            max_value = value;
            max_id = i;
        }
    }
        __shared__ float shared_max_value[256];
    __shared__ int64_t shared_max_id[256];

    shared_max_value[threadIdx.x] = max_value;
    shared_max_id[threadIdx.x] = max_id;
    __syncthreads();
    # pragma unroll
    for (int i = 256 / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            if (shared_max_value[threadIdx.x] < shared_max_value[threadIdx.x + i]) {
                shared_max_value[threadIdx.x] = shared_max_value[threadIdx.x + i];
                shared_max_id[threadIdx.x] = shared_max_id[threadIdx.x + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output0[out_id * in_reduce_size + in_id] = shared_max_id[0];
    }

}
extern void ArgMax_float_int64_t_cuda_ArgMax_84_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int64_t* output0) {
    ArgMax_float_int64_t_cuda_ArgMax_84<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	Reshape_86
// Description:	Reshape
// Input:
//	- name: Parameter_17_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Reshape_86_0	type: int64_t	shape: Shape{1}
extern "C" __launch_bounds__(256) __global__ void Reshape_int64_t_int64_t_cuda_Reshape_86(int64_t* input0, int64_t* output0)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1) { return; }
    output0[tid] = input0[tid];

}
extern void Reshape_int64_t_int64_t_cuda_Reshape_86_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int64_t* input0, int64_t* output0) {
    Reshape_int64_t_int64_t_cuda_Reshape_86<<<grids, blocks, mem, stream>>>(input0, output0);
}
// Node name:	 BlockFusion
// Input:
//	- name: Parameter_15_0	type: float	shape: Shape{50, 64, 3797}
//	- name: Parameter_17_0	type: int64_t	shape: Shape{}
//	- name: Constant_0_0	type: float	shape: Shape{3797, 256}
//	- name: Parameter_18_0	type: int64_t	shape: Shape{64}
//	- name: Constant_88_0	type: int64_t	shape: Shape{}
// Output:
//	- name: GatherV2_82_0	type: float	shape: Shape{64, 3797}
//	- name: GatherV2_19_0	type: float	shape: Shape{64, 256}
//	- name: id_1	type: int64_t	shape: Shape{}
// Fused functions:
// GatherV2_float_int64_t_float_cuda_GatherV2_82<<<dim3(950, 1, 1), dim3(256, 1, 1), 0, 0>>>(Parameter_15_0, Parameter_17_0, GatherV2_82_0);
// GatherV2_float_int64_t_float_cuda_GatherV2_19<<<dim3(64, 1, 1), dim3(256, 1, 1), 0, 0>>>(Constant_0_0, Parameter_18_0, GatherV2_19_0);
// Add_int64_t_int64_t_int64_t_cuda_Add_89<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, 0>>>(Parameter_17_0, Constant_88_0, id_1);
// Deduped function map: <src_function_name : deduped_function_name>

// Node name:	GatherV2_82
// Description:	GatherV2
// Input:
//	- name: Parameter_15_0	type: float	shape: Shape{50, 64, 3797}
//	- name: Parameter_17_0	type: int64_t	shape: Shape{}
// Output:
//	- name: GatherV2_82_0	type: float	shape: Shape{64, 3797}
__device__ __noinline__ void GatherV2_float_int64_t_float_cuda_GatherV2_82_block_kernel(float* input0, int64_t* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(256, 1, 1);
    const dim3 gridDim(950, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    float* params = input0;
    int64_t* indices = input1;
    float* out = output0;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 243008)
    {
        uint32_t batch_i = 0;
        uint32_t indices_i = 0;
        uint32_t slice_i = 0;
        indices_i = i / 243008;
        slice_i = i - indices_i * 243008;
        uint32_t gather_i = *(indices + indices_i);
        if (gather_i >= 50)
           out[i] = 0;
        else
        {
            uint32_t params_i = (batch_i * 50 + gather_i) * 243008 + slice_i;
            out[i] = __ldg(params + params_i);
        }
    }

}
// Node name:	GatherV2_19
// Description:	GatherV2
// Input:
//	- name: Constant_0_0	type: float	shape: Shape{3797, 256}
//	- name: Parameter_18_0	type: int64_t	shape: Shape{64}
// Output:
//	- name: GatherV2_19_0	type: float	shape: Shape{64, 256}
__device__ __noinline__ void GatherV2_float_int64_t_float_cuda_GatherV2_19_block_kernel(float* input0, int64_t* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 blockDim(256, 1, 1);
    const dim3 gridDim(64, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    float* params = input0;
    int64_t* indices = input1;
    float* out = output0;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 16384)
    {
        uint32_t batch_i = 0;
        uint32_t indices_i = 0;
        uint32_t slice_i = 0;
        indices_i = i / 256;
        slice_i = i - indices_i * 256;
        uint32_t gather_i = *(indices + indices_i);
        if (gather_i >= 3797)
           out[i] = 0;
        else
        {
            uint32_t params_i = (batch_i * 3797 + gather_i) * 256 + slice_i;
            out[i] = __ldg(params + params_i);
        }
    }

}
// Node name:	Add_89
// Description:	Add
// Input:
//	- name: Parameter_17_0	type: int64_t	shape: Shape{}
//	- name: Constant_88_0	type: int64_t	shape: Shape{}
// Output:
//	- name: id_1	type: int64_t	shape: Shape{}
__device__ __noinline__ void Add_int64_t_int64_t_int64_t_cuda_Add_89_block_kernel(int64_t* input0, int64_t* input1, int64_t* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 1){
        return;
    }
    const dim3 blockDim(1, 1, 1);
    const dim3 gridDim(1, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    output0[threadIdx.x] = add(input0[0], input1[0]);

}
extern "C" __global__  void BlockFusionKernel_float_int64_t_float_int64_t_int64_t_float_float_int64_t_cuda_GatherV2_GatherV2_Add_0(float* input0, int64_t* input1, float* input2, int64_t* input3, int64_t* input4, float* output0, float* output1, int64_t* output2)
{

    if ((int)blockIdx.x >= 0 && (int)blockIdx.x <= 949)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_82_block_kernel(input0, input1, output0, threadIdx.x, blockIdx.x - 0 + 0, NULL);
    }
    else if ((int)blockIdx.x >= 950 && (int)blockIdx.x <= 1013)
    {
        GatherV2_float_int64_t_float_cuda_GatherV2_19_block_kernel(input2, input3, output1, threadIdx.x, blockIdx.x - 950 + 0, NULL);
    }
    else if (blockIdx.x == 1014)
    {
        Add_int64_t_int64_t_int64_t_cuda_Add_89_block_kernel(input1, input4, output2, threadIdx.x, 0, NULL);
    }

}
extern void BlockFusionKernel_float_int64_t_float_int64_t_int64_t_float_float_int64_t_cuda_GatherV2_GatherV2_Add_0_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, int64_t* input1, float* input2, int64_t* input3, int64_t* input4, float* output0, float* output1, int64_t* output2) {
    BlockFusionKernel_float_int64_t_float_int64_t_int64_t_float_float_int64_t_cuda_GatherV2_GatherV2_Add_0<<<grids, blocks, mem, stream>>>(input0, input1, input2, input3, input4, output0, output1, output2);
}
// Node name:	Dot_77
// Description:	Dot
// Input:
//	- name: Reshape_76_0	type: float	shape: Shape{64, 256}
//	- name: Constant_11_0	type: float	shape: Shape{3797, 256}
// Output:
//	- name: Dot_77_0	type: float	shape: Shape{64, 3797}
extern "C" __global__ void Dot_float_float_float_cuda_Dot_77(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute)
{
    __shared__ float A_shared[256];
    __shared__ float B_shared[768];
    {
        {
          float compute_local[8];
          
          
          float A_shared_local[4];
          float B_shared_local[2];
          compute_local[(0)] = 0.000000e+00f;
          compute_local[(2)] = 0.000000e+00f;
          compute_local[(4)] = 0.000000e+00f;
          compute_local[(6)] = 0.000000e+00f;
          compute_local[(1)] = 0.000000e+00f;
          compute_local[(3)] = 0.000000e+00f;
          compute_local[(5)] = 0.000000e+00f;
          compute_local[(7)] = 0.000000e+00f;
          for (int k_outer = 0; k_outer < 32; ++k_outer) {
            __syncthreads();
            if (((int)threadIdx.x) < 256) {
              A_shared[(((int)threadIdx.x))] = A[((((((((int)blockIdx.x) / 40) * 8192) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)))];
            }
            B_shared[(((int)threadIdx.x))] = B[((((((((int)blockIdx.x) % 40) * 24576) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)))];
            if ((((((int)blockIdx.x) % 40) * 96) + (((int)threadIdx.x) >> 3)) < 3749) {
              B_shared[((((int)threadIdx.x) + 384))] = B[(((((((((int)blockIdx.x) % 40) * 24576) + ((((int)threadIdx.x) >> 3) * 256)) + (k_outer * 8)) + (((int)threadIdx.x) & 7)) + 12288))];
            }
            __syncthreads();
            for (int k_inner_outer = 0; k_inner_outer < 8; ++k_inner_outer) {
              A_shared_local[(0)] = A_shared[((((((int)threadIdx.x) / 48) * 8) + k_inner_outer))];
              A_shared_local[(1)] = A_shared[(((((((int)threadIdx.x) / 48) * 8) + k_inner_outer) + 64))];
              A_shared_local[(2)] = A_shared[(((((((int)threadIdx.x) / 48) * 8) + k_inner_outer) + 128))];
              A_shared_local[(3)] = A_shared[(((((((int)threadIdx.x) / 48) * 8) + k_inner_outer) + 192))];
              B_shared_local[(0)] = B_shared[((((((int)threadIdx.x) % 48) * 8) + k_inner_outer))];
              if ((((((int)blockIdx.x) % 40) * 96) + (((int)threadIdx.x) % 48)) < 3749) {
                B_shared_local[(1)] = B_shared[(((((((int)threadIdx.x) % 48) * 8) + k_inner_outer) + 384))];
              }
              compute_local[(0)] = (compute_local[(0)] + (A_shared_local[(0)] * B_shared_local[(0)]));
              compute_local[(2)] = (compute_local[(2)] + (A_shared_local[(1)] * B_shared_local[(0)]));
              compute_local[(4)] = (compute_local[(4)] + (A_shared_local[(2)] * B_shared_local[(0)]));
              compute_local[(6)] = (compute_local[(6)] + (A_shared_local[(3)] * B_shared_local[(0)]));
              if ((((((int)blockIdx.x) % 40) * 96) + (((int)threadIdx.x) % 48)) < 3749) {
                compute_local[(1)] = (compute_local[(1)] + (A_shared_local[(0)] * B_shared_local[(1)]));
                compute_local[(3)] = (compute_local[(3)] + (A_shared_local[(1)] * B_shared_local[(1)]));
                compute_local[(5)] = (compute_local[(5)] + (A_shared_local[(2)] * B_shared_local[(1)]));
                compute_local[(7)] = (compute_local[(7)] + (A_shared_local[(3)] * B_shared_local[(1)]));
              }
            }
          }
          compute[((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)))] = compute_local[(0)];
          compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 30376))] = compute_local[(2)];
          compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 60752))] = compute_local[(4)];
          compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 91128))] = compute_local[(6)];
          if ((((((int)blockIdx.x) % 40) * 96) + (((int)threadIdx.x) % 48)) < 3749) {
            compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 48))] = compute_local[(1)];
            compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 30424))] = compute_local[(3)];
            compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 60800))] = compute_local[(5)];
            compute[(((((((((int)blockIdx.x) / 40) * 121504) + ((((int)threadIdx.x) / 48) * 3797)) + ((((int)blockIdx.x) % 40) * 96)) + (((int)threadIdx.x) % 48)) + 91176))] = compute_local[(7)];
          }
        }


    }

}
extern void Dot_float_float_float_cuda_Dot_77_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
    Dot_float_float_float_cuda_Dot_77<<<grids, blocks, mem, stream>>>(A, B, compute);
}
// Node name:	Result_100
// Description:	Result
// Input:
//	- name: id_1	type: int64_t	shape: Shape{}
// Output:
//	- name: Result_100_0	type: int64_t	shape: Shape{}
void Result_int64_t_int64_t_cuda_lib_Result_100(cudaStream_t stream, int64_t* input0, int64_t* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,8, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	Constant_9
// Description:	Constant
// Input:
// Output:
//	- name: Constant_9_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_9(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_9_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_9_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_3
// Description:	Constant
// Input:
// Output:
//	- name: Constant_3_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_3(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_3_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_3_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_0
// Description:	Constant
// Input:
// Output:
//	- name: Constant_0_0	type: float	shape: Shape{3797, 256}
void Constant_float_cuda_Constant_0(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_0_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_0_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3888128];
    bin_file.read(tmp_mem, 3888128);
    cudaMemcpyAsync(output0, tmp_mem, 3888128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_12
// Description:	Constant
// Input:
// Output:
//	- name: Constant_12_0	type: float	shape: Shape{3797}
void Constant_float_cuda_Constant_12(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_12_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_12_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[15188];
    bin_file.read(tmp_mem, 15188);
    cudaMemcpyAsync(output0, tmp_mem, 15188, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_7
// Description:	Constant
// Input:
// Output:
//	- name: Constant_7_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_7(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_7_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_7_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_4
// Description:	Constant
// Input:
// Output:
//	- name: Constant_4_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_4(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_4_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_4_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_8
// Description:	Constant
// Input:
// Output:
//	- name: Constant_8_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_8(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_8_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_8_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_6
// Description:	Constant
// Input:
// Output:
//	- name: Constant_6_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_6(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_6_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_6_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_93
// Description:	Constant
// Input:
// Output:
//	- name: Constant_93_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_93(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_93_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_93_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Greater_92
// Description:	Greater
// Input:
//	- name: Sum_90_0	type: int64_t	shape: Shape{}
//	- name: Constant_91_0	type: int64_t	shape: Shape{}
// Output:
//	- name: Greater_92_0	type: char	shape: Shape{}
extern "C" __launch_bounds__(1) __global__ void Greater_int64_t_int64_t_char_cuda_Greater_92(int64_t* input0, int64_t* input1, char* output0)
{
    output0[threadIdx.x] = greater(input0[0], input1[0]);

}
extern void Greater_int64_t_int64_t_char_cuda_Greater_92_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, int64_t* input0, int64_t* input1, char* output0) {
    Greater_int64_t_int64_t_char_cuda_Greater_92<<<grids, blocks, mem, stream>>>(input0, input1, output0);
}
// Node name:	Constant_54
// Description:	Constant
// Input:
// Output:
//	- name: Constant_54_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_54(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_54_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_54_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_88
// Description:	Constant
// Input:
// Output:
//	- name: Constant_88_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_88(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_88_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_88_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_11
// Description:	Constant
// Input:
// Output:
//	- name: Constant_11_0	type: float	shape: Shape{3797, 256}
void Constant_float_cuda_Constant_11(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_11_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_11_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[3888128];
    bin_file.read(tmp_mem, 3888128);
    cudaMemcpyAsync(output0, tmp_mem, 3888128, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Result_98
// Description:	Result
// Input:
//	- name: tensor_87	type: int64_t	shape: Shape{50, 64}
// Output:
//	- name: Result_98_0	type: int64_t	shape: Shape{50, 64}
void Result_int64_t_int64_t_cuda_lib_Result_98(cudaStream_t stream, int64_t* input0, int64_t* output0)
{
    if (input0 != output0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,25600, cudaMemcpyDeviceToDevice, stream));
}
// Node name:	Constant_2
// Description:	Constant
// Input:
// Output:
//	- name: Constant_2_0	type: float	shape: Shape{4, 256, 256}
void Constant_float_cuda_Constant_2(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_2_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_2_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	 Elementwise Kernel Fusion
// Input:
//	- name: Constant_12_0	type: float	shape: Shape{3797}
//	- name: Dot_77_0	type: float	shape: Shape{64, 3797}
//	- name: GatherV2_82_0	type: float	shape: Shape{64, 3797}
// Output:
//	- name: Add_83_0	type: float	shape: Shape{64, 3797}
// Fused functions:
// Broadcast, Broadcast_78
// Add, Add_79
// Reshape, @tmp_37
// Add, output_1
extern "C" __launch_bounds__(256) __global__ void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Reshape_Add_1(float* input0, float* input1, float* input2, float* output0)
{
    for (int tid = blockIdx.x * 256 + threadIdx.x; tid < 243008; tid += 243200){
        float temp0 = input0[tid % 3797];
        float temp1 = add(input1[tid], temp0);
        float temp2 = add(temp1, input2[tid]);
        output0[tid] = temp2;
    }

}
extern void FusedKernel_float_float_float_float_cuda_Broadcast_Add_Reshape_Add_1_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, float* input0, float* input1, float* input2, float* output0) {
    FusedKernel_float_float_float_float_cuda_Broadcast_Add_Reshape_Add_1<<<grids, blocks, mem, stream>>>(input0, input1, input2, output0);
}
// Node name:	Constant_46
// Description:	Constant
// Input:
// Output:
//	- name: Constant_46_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_46(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_46_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_46_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_30
// Description:	Constant
// Input:
// Output:
//	- name: Constant_30_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_30(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_30_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_30_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_58
// Description:	Constant
// Input:
// Output:
//	- name: Constant_58_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_58(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_58_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_58_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_42
// Description:	Constant
// Input:
// Output:
//	- name: Constant_42_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_42(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_42_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_42_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_10
// Description:	Constant
// Input:
// Output:
//	- name: Constant_10_0	type: float	shape: Shape{256}
void Constant_float_cuda_Constant_10(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_10_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_10_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1024];
    bin_file.read(tmp_mem, 1024);
    cudaMemcpyAsync(output0, tmp_mem, 1024, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_34
// Description:	Constant
// Input:
// Output:
//	- name: Constant_34_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_34(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_34_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_34_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_1
// Description:	Constant
// Input:
// Output:
//	- name: Constant_1_0	type: float	shape: Shape{4, 256, 256}
void Constant_float_cuda_Constant_1(cudaStream_t stream, float* output0)
{
    std::ifstream bin_file("./Constant/Constant_1_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_1_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[1048576];
    bin_file.read(tmp_mem, 1048576);
    cudaMemcpyAsync(output0, tmp_mem, 1048576, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}

#ifndef __NNFUSION_GRAPH_CONFIG__
#define __NNFUSION_GRAPH_CONFIG__
#define NNFUSION_GRAPH_INPUT_NUM 6
#define NNFUSION_GRAPH_OUTPUT_NUM 6
#define NNFUSION_GRAPH_INPUT_DTYPE_0 float
#define NNFUSION_GRAPH_INPUT_SHAPE_0 {64, 256}
#define NNFUSION_GRAPH_INPUT_DTYPE_1 float
#define NNFUSION_GRAPH_INPUT_SHAPE_1 {64, 256}
#define NNFUSION_GRAPH_INPUT_DTYPE_2 float
#define NNFUSION_GRAPH_INPUT_SHAPE_2 {50, 64, 3797}
#define NNFUSION_GRAPH_INPUT_DTYPE_3 int64_t
#define NNFUSION_GRAPH_INPUT_SHAPE_3 {50, 64}
#define NNFUSION_GRAPH_INPUT_DTYPE_4 int64_t
#define NNFUSION_GRAPH_INPUT_SHAPE_4 {}
#define NNFUSION_GRAPH_INPUT_DTYPE_5 int64_t
#define NNFUSION_GRAPH_INPUT_SHAPE_5 {64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_0 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_0 {64, 256}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_1 float
#define NNFUSION_GRAPH_OUTPUT_SHAPE_1 {64, 256}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_2 int64_t
#define NNFUSION_GRAPH_OUTPUT_SHAPE_2 {50, 64}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_3 char
#define NNFUSION_GRAPH_OUTPUT_SHAPE_3 {}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_4 int64_t
#define NNFUSION_GRAPH_OUTPUT_SHAPE_4 {}
#define NNFUSION_GRAPH_OUTPUT_DTYPE_5 int64_t
#define NNFUSION_GRAPH_OUTPUT_SHAPE_5 {64}
#endif

// Node name:	Constant_38
// Description:	Constant
// Input:
// Output:
//	- name: Constant_38_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_38(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_38_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_38_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// Node name:	Constant_91
// Description:	Constant
// Input:
// Output:
//	- name: Constant_91_0	type: int64_t	shape: Shape{}
void Constant_int64_t_cuda_Constant_91(cudaStream_t stream, int64_t* output0)
{
    std::ifstream bin_file("./Constant/Constant_91_0.bin" , std::ios::in | std::ios::binary);
    if(bin_file.fail())
    {
    	printf("Load Constant_91_0 failed.\n");
    	exit(1);
    }
    char* tmp_mem = new char[8];
    bin_file.read(tmp_mem, 8);
    cudaMemcpyAsync(output0, tmp_mem, 8, cudaMemcpyHostToDevice, stream);
    bin_file.close();

}
// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN
int get_device_type()
{
    return 0;
}

extern "C" void cuda_init()
{
//CUDA_SAFE_CALL(cudaDeviceReset());
// total memory:14125248
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_0_CUDA_GPU0_allocator_memory_pool,4096000));
CUDA_SAFE_CALL(cudaMemset((void*)group_0_CUDA_GPU0_allocator_memory_pool, 0, 4096000));
GatherV2_19_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
GatherV2_82_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+65536);
Reshape_22_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+0);
Reshape_21_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1037568);
Reshape_26_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1037568);
BatchMatMul_24_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1103104);
BatchMatMul_28_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1365248);
Reshape_25_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1103104);
Reshape_29_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1365248);
GatherV2_55_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1627392);
GatherV2_47_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1692928);
GatherV2_39_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1758464);
GatherV2_31_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1824000);
GatherV2_59_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1889536);
GatherV2_51_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+1955072);
GatherV2_43_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2020608);
GatherV2_35_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2086144);
Dot_77_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+2151680);
Add_83_0 = (float*)(group_0_CUDA_GPU0_allocator_memory_pool+3123712);
Less_94_0 = (char*)(group_0_CUDA_GPU0_allocator_memory_pool+4095744);
Reshape_86_0 = (int64_t*)(group_0_CUDA_GPU0_allocator_memory_pool+4095808);
Sum_90_0 = (int64_t*)(group_0_CUDA_GPU0_allocator_memory_pool+4095872);
Greater_92_0 = (char*)(group_0_CUDA_GPU0_allocator_memory_pool+4095936);
CUDA_SAFE_CALL(cudaSetDevice(0));
CUDA_SAFE_CALL(cudaMalloc((void**)&group_persist_CUDA_GPU0_allocator_memory_pool,10029248));
CUDA_SAFE_CALL(cudaMemset((void*)group_persist_CUDA_GPU0_allocator_memory_pool, 0, 10029248));
Constant_88_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+0);
Constant_0_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+64);
id_1 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+3888192);
Constant_11_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+3888256);
Constant_54_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+7776384);
Constant_1_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7776448);
Reshape_23_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+7776448);
Constant_2_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8825024);
Reshape_27_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+8825024);
Constant_46_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873600);
Constant_38_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873664);
Constant_30_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873728);
Constant_58_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873792);
Constant_50_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873856);
Constant_42_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873920);
Constant_34_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+9873984);
Constant_9_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9874048);
Constant_10_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9875072);
Constant_5_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9876096);
Constant_6_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9877120);
Constant_3_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9878144);
Constant_4_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9879168);
Constant_7_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9880192);
Constant_8_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9881216);
h_2 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9882240);
c_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9947776);
Reshape_76_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9882240);
Constant_12_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+10013312);
tensor_84 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+10028544);
Result_101_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+10028544);
Result_100_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+3888192);
Constant_93_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+10029056);
Constant_91_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+10029120);
// tensor_87 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
cond_0 = (char*)(group_persist_CUDA_GPU0_allocator_memory_pool+10029184);
Result_99_0 = (char*)(group_persist_CUDA_GPU0_allocator_memory_pool+10029184);
// Result_98_0 = (int64_t*)(group_persist_CUDA_GPU0_allocator_memory_pool+18446744073709551615);
Result_97_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9947776);
Result_96_0 = (float*)(group_persist_CUDA_GPU0_allocator_memory_pool+9882240);
// create streams/handles
 // name=@tmp_42
Constant_int64_t_cuda_Constant_88(0, Constant_88_0);
 // name=embedding_0_weight
Constant_float_cuda_Constant_0(0, Constant_0_0);
 // name=out_0_weight
Constant_float_cuda_Constant_11(0, Constant_11_0);
 // name=@tmp_22
Constant_int64_t_cuda_Constant_54(0, Constant_54_0);
 // name=weight_ih_l0_t_0
Constant_float_cuda_Constant_1(0, Constant_1_0);
 // name=weight_hh_l0_t_0
Constant_float_cuda_Constant_2(0, Constant_2_0);
 // name=@tmp_16
Constant_int64_t_cuda_Constant_46(0, Constant_46_0);
 // name=@tmp_10
Constant_int64_t_cuda_Constant_38(0, Constant_38_0);
 // name=@tmp_4
Constant_int64_t_cuda_Constant_30(0, Constant_30_0);
 // name=@tmp_25
Constant_int64_t_cuda_Constant_58(0, Constant_58_0);
 // name=@tmp_19
Constant_int64_t_cuda_Constant_50(0, Constant_50_0);
 // name=@tmp_13
Constant_int64_t_cuda_Constant_42(0, Constant_42_0);
 // name=@tmp_7
Constant_int64_t_cuda_Constant_34(0, Constant_34_0);
 // name=bias_ih_3_0
Constant_float_cuda_Constant_9(0, Constant_9_0);
 // name=bias_hh_3_0
Constant_float_cuda_Constant_10(0, Constant_10_0);
 // name=bias_ih_1_0
Constant_float_cuda_Constant_5(0, Constant_5_0);
 // name=bias_hh_1_0
Constant_float_cuda_Constant_6(0, Constant_6_0);
 // name=bias_ih_0_0
Constant_float_cuda_Constant_3(0, Constant_3_0);
 // name=bias_hh_0_0
Constant_float_cuda_Constant_4(0, Constant_4_0);
 // name=bias_ih_2_0
Constant_float_cuda_Constant_7(0, Constant_7_0);
 // name=bias_hh_2_0
Constant_float_cuda_Constant_8(0, Constant_8_0);
 // name=out_0_bias
Constant_float_cuda_Constant_12(0, Constant_12_0);
 // name=@tmp_46
Constant_int64_t_cuda_Constant_93(0, Constant_93_0);
 // name=@tmp_44
Constant_int64_t_cuda_Constant_91(0, Constant_91_0);
CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0));
}


extern "C" int kernel_entry(float* Parameter_13_0, float* Parameter_14_0, float* Parameter_15_0, int64_t* Parameter_16_0, int64_t* Parameter_17_0, int64_t* Parameter_18_0, float* Result_96_0, float* Result_97_0, int64_t* Result_98_0, char* Result_99_0, int64_t* Result_100_0, int64_t* Result_101_0)
{
// kernel_entry_init
 // name=blockfusion_kernel_104
BlockFusionKernel_float_int64_t_float_int64_t_int64_t_float_float_int64_t_cuda_GatherV2_GatherV2_Add_0_Call(dim3(1015, 1, 1), dim3(256, 1, 1), 0, 0, Parameter_15_0, Parameter_17_0, Constant_0_0, Parameter_18_0, Constant_88_0, GatherV2_82_0, GatherV2_19_0, Result_100_0);
 // name=Reshape_23
// eliminated: Reshape_float_float_cuda_Reshape_23_Call(dim3(1024, 1, 1), dim3(256, 1, 1), 0, 0, Constant_1_0, Reshape_23_0);
 // name=Reshape_22
// eliminated: Reshape_float_float_cuda_Reshape_22_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, GatherV2_19_0, Reshape_22_0);
 // name=Reshape_27
// eliminated: Reshape_float_float_cuda_Reshape_27_Call(dim3(1024, 1, 1), dim3(256, 1, 1), 0, 0, Constant_2_0, Reshape_27_0);
 // name=h_0
Reshape_float_float_cuda_Reshape_21_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, Parameter_13_0, Reshape_21_0);
 // name=Reshape_26
// eliminated: Reshape_float_float_cuda_Reshape_21_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, Reshape_21_0, Reshape_26_0);
 // name=blockfusion_kernel_105
BlockFusionKernel_float_float_float_float_float_float_cuda_BatchMatMul_BatchMatMul_1_Call(dim3(16, 1, 1), dim3(256, 1, 1), 0, 0, Reshape_26_0, Reshape_27_0, Reshape_22_0, Reshape_23_0, BatchMatMul_28_0, BatchMatMul_24_0);
 // name=Reshape_25
// eliminated: Reshape_float_float_cuda_Reshape_25_Call(dim3(256, 1, 1), dim3(256, 1, 1), 0, 0, BatchMatMul_24_0, Reshape_25_0);
 // name=Reshape_29
// eliminated: Reshape_float_float_cuda_Reshape_29_Call(dim3(256, 1, 1), dim3(256, 1, 1), 0, 0, BatchMatMul_28_0, Reshape_29_0);
 // name=blockfusion_kernel_106
BlockFusionKernel_float_int64_t_int64_t_int64_t_int64_t_float_int64_t_int64_t_int64_t_int64_t_float_float_float_float_float_float_float_float_cuda_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_GatherV2_2_Call(dim3(512, 1, 1), dim3(256, 1, 1), 0, 0, Reshape_29_0, Constant_34_0, Constant_42_0, Constant_50_0, Constant_58_0, Reshape_25_0, Constant_30_0, Constant_38_0, Constant_46_0, Constant_54_0, GatherV2_35_0, GatherV2_43_0, GatherV2_51_0, GatherV2_59_0, GatherV2_31_0, GatherV2_39_0, GatherV2_47_0, GatherV2_55_0);
 // name=ElementWiseFused_102
FusedKernel_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_float_cuda_Broadcast_Add_Broadcast_Add_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_Add_Tanh_Broadcast_Add_Broadcast_Add_Add_Sigmoid_Multiply_0_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, Constant_8_0, GatherV2_51_0, Constant_7_0, GatherV2_47_0, Constant_4_0, GatherV2_35_0, Constant_3_0, GatherV2_31_0, Constant_6_0, GatherV2_43_0, Constant_5_0, GatherV2_39_0, Parameter_14_0, Constant_10_0, GatherV2_59_0, Constant_9_0, GatherV2_55_0, Result_97_0, Result_96_0);
 // name=@tmp_33
Reshape_float_float_cuda_Reshape_21_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, Result_96_0, Reshape_76_0);
 // name=Dot_77
Dot_float_float_float_cuda_Dot_77_Call(dim3(80, 1, 1), dim3(384, 1, 1), 0, 0, Reshape_76_0, Constant_11_0, Dot_77_0);
 // name=ElementWiseFused_103
FusedKernel_float_float_float_float_cuda_Broadcast_Add_Reshape_Add_1_Call(dim3(950, 1, 1), dim3(256, 1, 1), 0, 0, Constant_12_0, Dot_77_0, GatherV2_82_0, Add_83_0);
 // name=output_2
ArgMax_float_int64_t_cuda_ArgMax_84_Call(dim3(64, 1, 1), dim3(256, 1, 1), 0, 0, Add_83_0, Result_101_0);
 // name=Result_101
// eliminated (extern_result_memory): Result_int64_t_int64_t_cuda_lib_Result_101(0, Result_101_0, Result_101_0);
 // name=Result_100
// eliminated (extern_result_memory): Result_int64_t_int64_t_cuda_lib_Result_100(0, Result_100_0, Result_100_0);
 // name=@tmp_47
Less_int64_t_int64_t_char_cuda_Less_94_Call(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Result_100_0, Constant_93_0, Less_94_0);
 // name=@tmp_40
Reshape_int64_t_int64_t_cuda_Reshape_86_Call(dim3(1, 1, 1), dim3(256, 1, 1), 0, 0, Parameter_17_0, Reshape_86_0);
 // name=blockfusion_kernel_107
tensor_87 = Parameter_16_0;
/* memref */BlockFusionKernel_int64_t_int64_t_int64_t_int64_t_int64_t_cuda_ScatterND_Sum_3_Call(dim3(2, 1, 1), dim3(256, 1, 1), 0, 0, Parameter_16_0, Reshape_86_0, Result_101_0, Result_98_0, Sum_90_0);
 // name=@tmp_45
Greater_int64_t_int64_t_char_cuda_Greater_92_Call(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Sum_90_0, Constant_91_0, Greater_92_0);
 // name=cond_0
And_char_char_char_cuda_And_95_Call(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Greater_92_0, Less_94_0, Result_99_0);
 // name=Result_99
// eliminated (extern_result_memory): Result_char_char_cuda_lib_Result_99(0, Result_99_0, Result_99_0);
 // name=Result_98
// eliminated (extern_result_memory): Result_int64_t_int64_t_cuda_lib_Result_98(0, Result_98_0, Result_98_0);
 // name=Result_97
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_97(0, Result_97_0, Result_97_0);
 // name=Result_96
// eliminated (extern_result_memory): Result_float_float_cuda_lib_Result_97(0, Result_96_0, Result_96_0);
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

