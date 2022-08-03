#include <iostream>
#include <unordered_map>
#include <stdio.h>
#include "runtime.h"
#include <sstream>
#include <windows.h>
#include "D3D12APIWrapper.h"
void* group_persist_HLSL0_allocator_memory_pool;
void* Parameter_1_0;
void* Parameter_0_0;
void* tensor_2;
void* Result_3_0;

int dxModuleLaunchAsync(void* hModule, std::string* kernel_names, void*** args, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        void* shader = dxModuleGetShader(hModule, kernel_names[i].c_str());
        dxShaderLaunchAsync(shader, args[i], 0);
    }
    return 0;
};

void* Power_int64_t_int64_t_int64_t_hlsl_Power_2_module;
// 0: CUDA_GPU; 1: ROCM_GPU; 2: GENERIC_CPU; 3: HLSL; 4: GraphCore; 5: UNKNOWN
int get_device_type()
{
    return 3;
}

void hlsl_init()
{
    dxModuleSetCompat("cs_6_2");
// total memory:192
group_persist_HLSL0_allocator_memory_pool = dxMemAlloc(192);
Parameter_1_0 = (char*)group_persist_HLSL0_allocator_memory_pool + 0;
Parameter_0_0 = (char*)group_persist_HLSL0_allocator_memory_pool + 64;
tensor_2 = (char*)group_persist_HLSL0_allocator_memory_pool + 128;
Result_3_0 = (char*)group_persist_HLSL0_allocator_memory_pool + 128;
// create streams
Power_int64_t_int64_t_int64_t_hlsl_Power_2_module = dxModuleLoad("file://HLSL/Power_int64_t_int64_t_int64_t_hlsl_Power_2.hlsl");
auto Power_int64_t_int64_t_int64_t_hlsl_Power_2_module_dict = *(std::unordered_map<std::string, void*>*)Power_int64_t_int64_t_int64_t_hlsl_Power_2_module;
for (auto& p : Power_int64_t_int64_t_int64_t_hlsl_Power_2_module_dict)
{
    if (!p.second) {
        std::cout << "Invalid Shader Source for Compilation: file://HLSL/Power_int64_t_int64_t_int64_t_hlsl_Power_2.hlsl";
        exit(1);
    }
}
dxStreamSynchronize(0);
}

int get_workspace_size()
{
    return 192;
}

int kernel_entry(void* Parameter_0_0, void* Parameter_1_0, void* Result_3_0)
{
std::string Power_int64_t_int64_t_int64_t_hlsl_Power_2_kernel_names[] = { "template_op_kernel0" };
void* Power_int64_t_int64_t_int64_t_hlsl_Power_2_module_args_0[] = { Parameter_0_0, Parameter_1_0, tensor_2 };
void** Power_int64_t_int64_t_int64_t_hlsl_Power_2_args[] = { Power_int64_t_int64_t_int64_t_hlsl_Power_2_module_args_0 };
dxModuleLaunchAsync(Power_int64_t_int64_t_int64_t_hlsl_Power_2_module, Power_int64_t_int64_t_int64_t_hlsl_Power_2_kernel_names, Power_int64_t_int64_t_int64_t_hlsl_Power_2_args, std::size(Power_int64_t_int64_t_int64_t_hlsl_Power_2_args));
// Result_int64_t_int64_t_hlsl_Result_3(tensor_2, Result_3_0);
dxMemcpyDtoDAsync(Result_3_0, tensor_2, sizeof(int64_t) * 3, nullptr);
return 0;
}


//int kernel_entry_host(void* Parameter_0_0_host, void* Parameter_1_0_host, void* Result_3_0_host)
//{
//dxMemcpyHtoDAsync(Parameter_0_0, Parameter_0_0_host, sizeof(int64_t) * 3, 0);
//dxMemcpyHtoDAsync(Parameter_1_0, Parameter_1_0_host, sizeof(int64_t) * 3, 0);
//kernel_entry(Parameter_0_0, Parameter_1_0, Result_3_0);
//dxMemcpyDtoHAsync(Result_3_0_host, Result_3_0, sizeof(int64_t) * 3, nullptr);
//dxStreamSynchronize(0);
//return 0;
//}


void hlsl_free()
{
dxMemFree(group_persist_HLSL0_allocator_memory_pool);
dxFinalize();
}

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

