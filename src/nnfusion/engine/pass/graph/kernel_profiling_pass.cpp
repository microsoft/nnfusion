// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_profiling_pass.hpp"
#include "nnfusion/engine/profiler/cuda_runtime.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;
DEFINE_bool(fenable_kernel_profiling, false, "profile kernel time.");
DECLARE_string(fstream_assign_policy);
DEFINE_bool(fmerge_prof_compiling, false, "");

bool KernelProfilingPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fenable_kernel_profiling && FLAGS_fstream_assign_policy != "kernel_prof_based")
        return true;
    if (FLAGS_fmerge_prof_compiling)
        merged_profiling_pass(graph);
    else
        default_profiling_pass(graph);
    return true;
}

bool KernelProfilingPass::default_profiling_pass(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if ((*it)["Kernel_Selection_Result"].is_valid())
        {
            auto kernel_result = (*it)["Kernel_Selection_Result"]
                                     .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            auto n_device_type = kernel_result.first;
            auto kernel = kernel_result.second;

            if (!(*it)["Kernel_Profiling_Result"].is_valid() && !it->is_constant())
            {
                auto profiling_kernel =
                    [](KernelEmitter::Pointer kernel,
                       IProfilingRuntime::Pointer runtime) -> KernelProfilingRecord::Pointer {
                    if (kernel->get_or_emit_source())
                    {
                        auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                        nnfusion::profiler::Profiler prof(runtime, pctx);

                        if (prof.execute())
                        {
                            double kernel_time = pctx->result.get_device_avg();
                            auto record = make_shared<KernelProfilingRecord>();
                            record->kernel_time_in_us = kernel_time;
                            record->valid = true;

                            NNFUSION_LOG(INFO)
                                << "Profiling kernel: " << kernel->get_function_name()
                                << ", kernel time(us):" << kernel_time;
                            return record;
                        }
                        else
                        {
                            NNFUSION_LOG(INFO) << "Kernel Failed.";
                        }
                    }
                    return nullptr;
                };
                KernelProfilingRecord::Pointer result;

                if (n_device_type == CUDA_GPU)
                {
                    result = profiling_kernel(kernel, CUPTIRuntime::Runtime());
                }
                else if (n_device_type == GENERIC_CPU)
                {
                    result = profiling_kernel(kernel, CPUDefaultRuntime::Runtime());
                }
                else
                {
                    result = profiling_kernel(kernel, get_default_runtime(n_device_type));
                    if (!result && n_device_type == ROCM_GPU)
                    {
                        result = profiling_kernel(kernel, get_default_runtime(CUDA_GPU));
                    }
                }

                if (result)
                {
                    (*it)["Kernel_Profiling_Result"] = result;
                }
            }
        }
    }
    return true;
}

bool KernelProfilingPass::merged_profiling_pass(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    std::unordered_map<KernelEmitter::Pointer, ProfilingContext::Pointer> kernel_pctx;
    for (auto it : nodes)
    {
        if ((*it)["Kernel_Selection_Result"].is_valid())
        {
            auto kernel_result = (*it)["Kernel_Selection_Result"]
                                     .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            auto n_device_type = kernel_result.first;
            auto kernel = kernel_result.second;

            if (!(*it)["Kernel_Profiling_Result"].is_valid() && !it->is_constant())
            {
                if (n_device_type == GENERIC_CPU)
                {
                    auto runtime = CPUDefaultRuntime::Runtime();
                    auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                    kernel_pctx[kernel] = pctx;
                    NNFUSION_CHECK(runtime->codegen(pctx));
                }
            }
        }
    }

    auto runtime = CPUDefaultRuntime::Runtime();
    NNFUSION_CHECK(runtime->general_compile());

    for (auto it : nodes)
    {
        if ((*it)["Kernel_Selection_Result"].is_valid())
        {
            auto kernel_result = (*it)["Kernel_Selection_Result"]
                                     .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            auto n_device_type = kernel_result.first;
            auto kernel = kernel_result.second;

            if (kernel_pctx.find(kernel) != kernel_pctx.end())
            {
                KernelProfilingRecord::Pointer result;
                auto profiling_kernel = [&kernel_pctx](
                    KernelEmitter::Pointer kernel,
                    IProfilingRuntime::Pointer runtime) -> KernelProfilingRecord::Pointer {
                    if (kernel->get_or_emit_source())
                    {
                        auto pctx = kernel_pctx[kernel];
                        nnfusion::profiler::Profiler prof(runtime, pctx);

                        if (prof.execute())
                        {
                            double kernel_time = pctx->result.get_device_avg();
                            auto record = make_shared<KernelProfilingRecord>();
                            record->kernel_time_in_us = kernel_time;
                            record->valid = true;

                            NNFUSION_LOG(INFO)
                                << "Profiling kernel: " << kernel->get_function_name()
                                << ", kernel time(us):" << kernel_time;
                            return record;
                        }
                        else
                        {
                            NNFUSION_LOG(INFO) << "Kernel Failed.";
                        }
                    }
                    return nullptr;
                };

                if (n_device_type == CUDA_GPU)
                {
                    result = profiling_kernel(kernel, CUPTIRuntime::Runtime());
                }
                else if (n_device_type == GENERIC_CPU)
                {
                    result = profiling_kernel(kernel, CPUDefaultRuntime::Runtime());
                }
                else
                {
                    result = profiling_kernel(kernel, get_default_runtime(n_device_type));
                    if (!result && n_device_type == ROCM_GPU)
                    {
                        result = profiling_kernel(kernel, get_default_runtime(CUDA_GPU));
                    }
                }

                if (result)
                {
                    (*it)["Kernel_Profiling_Result"] = result;
                }
            }
        }
    }
    return true;
}