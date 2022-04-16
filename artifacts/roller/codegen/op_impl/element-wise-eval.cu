#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>
#include "element-wise.cuh"

int N = 8192;
int M = 8192;

int main(int argc, char *argv[])
{
        std::string path;
        for (int i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-n") == 0) N = atoi(argv[i+1]);
                if (strcmp(argv[i], "-m") == 0) M = atoi(argv[i+1]);
                if (strcmp(argv[i], "-p") == 0) path = argv[i+1];
        }
        std::string code_path = path + "/my_kernel.cc";
        std::string mod_path = path + "/my_kernel.out";
        int input_size0 = N * M;
        int input_size1 = N * M;
        int output_size = N * M;

        checkCudaErrors(cuInit(0));
        CUdevice device;
        checkCudaErrors(cuDeviceGet(&device, 0));
        CUcontext context;
        checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

        CUmodule cuModule_;
        CUfunction cuda_func_;

        float *Ah, *Bh, *Ch;
        CUdeviceptr Ad, Bd, Cd;
        Ah = (float*)malloc(input_size0 * sizeof(float));
        Bh = (float*)malloc(input_size1 * sizeof(float));
        Ch = (float*)malloc(output_size * sizeof(float));

        cudaMalloc((void **)&Ad, input_size0 * sizeof(float));
        cudaMalloc((void **)&Bd, input_size1 * sizeof(float));
        cudaMalloc((void **)&Cd, output_size * sizeof(float));

        checkCudaErrors(cuMemAlloc(&Ad, sizeof(float) * input_size0));
        checkCudaErrors(cuMemAlloc(&Bd, sizeof(float) * input_size1));
        checkCudaErrors(cuMemAlloc(&Cd, sizeof(float) * output_size));

        void* param[] = {&Ad, &Bd, &Cd};

        srand(1);
        for (int i = 0; i < input_size0; ++ i) Ah[i] = rand();
        for (int i = 0; i < input_size1; ++ i) Bh[i] = rand();

        checkCudaErrors(cuMemcpyHtoD(Ad, Ah, input_size0 * sizeof(float)));
        checkCudaErrors(cuMemcpyHtoD(Bd, Bh, input_size1 * sizeof(float)));

        // checkCudaErrors(cuModuleLoad(&cuModule_, mod_path.c_str()));
        // checkCudaErrors(cuModuleGetFunction(&cuda_func_, cuModule_, "template_op_kernel0"));
        // FILE* fp = fopen(code_path.c_str(), "r");
        // int block_x = 1, block_y = 1, thread_x = 1, thread_y = 1;
        // while (!feof(fp)) {
        //         char *line;
        //         line = (char*)malloc(2000 * sizeof(char));
        //         fgets(line, 2000, fp);
        //         std::string std_line = std::string(line);
        //         if (int(std_line.find("[thread_extent] blockIdx.x")) > -1)
        //         {
        //                 int k = std_line.rfind("=");
        //                 block_x = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
        //         }
        //         if (int(std_line.find("[thread_extent] blockIdx.y")) > -1)
        //         {
        //                         int k = std_line.rfind("=");
        //                 block_y = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
        //         }
        //         if (int(std_line.find("[thread_extent] threadIdx.x")) > -1)
        //         {
        //                 int k = std_line.rfind("=");
        //                 thread_x = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
        //         }
        //         if (int(std_line.find("[thread_extent] threadIdx.y")) > -1)
        //         {
        //                 int k = std_line.rfind("=");
        //                 thread_y = std::atoi(std_line.substr(k + 2, std_line.length() - k).c_str());
        //         }
        // }
        // printf("path: %s Grid: (%d %d, 1) Block: (%d %d 1)\n", path.c_str(), block_x, block_y, thread_x, thread_y);
        // for (int i = 0; i < 10; ++ i)
        // {
        //         checkCudaErrors(cuLaunchKernel(cuda_func_, block_x, block_y, 1, thread_x, thread_y, 1, 0, 0, (void**) param, 0));
        //         cudaDeviceSynchronize();
        // }

        dim3 grid(N / 32 * M / 32, 1, 1);
        dim3 block(32 * 32, 1, 1);
        for (int i = 0; i < 10; ++i)
        {
                default_function_kernel0<<<grid, block>>>((float *)Cd, (float *)Ad, (float *)Bd);
                cudaDeviceSynchronize();
        }
}