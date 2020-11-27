// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Use this Profiler to run each operator
 * \author wenxh
 * \todo This profiler only support linux since it will invoke native commands.
 */

#include "profiler.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

#include <chrono>
#include <ctime>
#include <ratio>

using namespace nnfusion;
using namespace nnfusion::profiler;
using namespace std::chrono;

DECLARE_bool(fmerge_prof_compiling);

Profiler::Profiler(IProfilingRuntime::Pointer rt, ProfilingContext::Pointer context)
{
    this->rt = rt;
    this->pctx = context;
    ///\todo: verify if the runtime is ok
}

double Profiler::execute(void** input, void** output)
{
    if (rt == nullptr)
        return -1.0;

    for (int i = 0; i < pctx->host_times; i++)
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        double device_time_span;
        if (FLAGS_fmerge_prof_compiling)
        {
            if (auto cpu_rt = dynamic_pointer_cast<CPUDefaultRuntime>(rt))
                device_time_span = cpu_rt->sep_invoke(this->pctx, input, output);
        }
        else
            device_time_span = rt->execute(this->pctx, input, output);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        if (device_time_span < 0)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel launch failed.";
            continue;
        }
        pctx->result.record_host_duration(time_span.count());
        pctx->result.record_device_duration(device_time_span);
    }
    pctx->result.set_ready();
    return pctx->result.get_device_avg();
}

bool Profiler::execute()
{
    auto& kernel_mem = pctx->kernel_memory;
    bool ret = execute(kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) >= 0;
    return ret;
}

bool Profiler::find_best()
{
    return false;
}

bool Profiler::execute_all()
{
    return false;
}

void GraphEvaluate::create_profiling_contexts(shared_ptr<GNode> gnode)
{
    if (gnode->get_op_ptr()->is_tensor_op())
    {
        return;
    }
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), dev_type, element::f32);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    for (auto kernel_reg : kernel_regs)
    {
        //if (kernel_reg->m_tag != "reference")
        //    continue;
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            // Replacing the kernel;
            auto pctx = make_shared<ProfilingContext>(kernel);
            this->gctx.set_profiling_context(gnode, pctx);
            return;
        }
    }

    NNFUSION_LOG(ERROR) << "Invalid reference kernel for " << gnode->get_name()
                        << " (op type : " << gnode->get_op_type() << ").";
}

IProfilingRuntime::Pointer nnfusion::profiler::get_default_runtime(NNFusion_DeviceType dev_t)
{
    IProfilingRuntime::Pointer ip = nullptr;
    switch (dev_t)
    {
    case CUDA_GPU: ip = CudaDefaultRuntime::Runtime(); break;
    case ROCM_GPU: ip = RocmDefaultRuntime::Runtime(); break;
    case GENERIC_CPU: ip = ReferenceRuntime::Runtime(); break;
    }
    if (ip != nullptr && ip->check_env())
        return ip;
    return nullptr;
}

IProfilingRuntime::Pointer nnfusion::profiler::get_default_runtime(string dev_str)
{
    return get_default_runtime(get_device_type(dev_str));
}