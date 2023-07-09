// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Profiler::RocmRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#include "rocm_runtime.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"

using namespace nnfusion::profiler;
using namespace nnfusion::kernels;

bool RocmDefaultRuntime::check_env()
{
    return file_exists("/opt/rocm/bin/hipcc");
}

bool RocmDefaultRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(nnfusion::tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = ke->source_code->get_symbol();

    int ret =
        system(("/opt/rocm/bin/hipcc\t-O3\t-Wno-ignored-attributes\t-fPIC\t-lMIOpen\t-lrocblas\t-I/"
                "opt/rocm/include\t-L/opt/rocm/lib\t-I/opt/rocm/rocblas/include\t-I/opt/rocm/"
                "rocrand/include\t-I/"
                "opt/rocm/hiprand/include\t-I/opt/rocm/hipsparse/include\t--shared\t" +
                srcname + "\t-o\t" + objname)
                   .c_str());
    if (ret != 0)
        return false;
    if (!file_exists(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_function_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

bool RocmDefaultRuntime::hipfy(const ProfilingContext::Pointer& ke)
{
    auto cudafile = ke->source_code;

    if (ke->entry_point != nullptr)
        return true;
    string filename = string(nnfusion::tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string cu_srcname = filename + ".cu";
    string rocm_srcname = filename + ".cpp";

    ofstream source_file(cu_srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    // Save file from result
    auto hipify_exec = nnfusion::codegen::get_file_from_templates("rocm_adapter/hipify-nnfusion");

    // hipfy the file
    int ret =
        system((hipify_exec + " " + cu_srcname +
                " | grep -v 'include.*cublas_v2' | grep -v 'include.*cudnn' > " + rocm_srcname)
                   .c_str());

    if (ret != 0)
        return false;
    // Replacing Header Files
    ret = system(("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' " + rocm_srcname +
                  " && sed -i "
                  "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' " +
                  rocm_srcname)
                     .c_str());

    if (ret != 0)
        return false;
    NNFUSION_LOG(INFO) << "Cuda file is:" << cu_srcname;
    NNFUSION_LOG(INFO) << "ROCM file is:" << rocm_srcname;
    // Save rocm_adapter.h for the file.
    auto rocmadapter = rocm_srcname.substr(0, rocm_srcname.find_last_of("/\\")) + "/rocm_adapter.h";
    nnfusion::codegen::copy_file_from_templates("rocm_adapter/rocm_adapter.h", rocmadapter);

    ke->source_code->change_symbol(rocm_srcname);
    return true;
}

double RocmDefaultRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    // Reuse cuda codegen.
    if (codegen(ke) == false)
        return -1.0;
    // This step may looks more like an option in future.
    if (hipfy(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

RocmDefaultRuntime::Pointer RocmDefaultRuntime::Runtime()
{
    static RocmDefaultRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
    {
        predefined = make_shared<RocmDefaultRuntime>();
        predefined->set_dt(ROCM_GPU);
    }
    return predefined;
}
