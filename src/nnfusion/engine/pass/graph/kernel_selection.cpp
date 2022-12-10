// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_selection.hpp"
#include <queue>
#include <utility>
#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

DEFINE_bool(fkernel_selection, false, "Select 'best' kernel based on the profiling information.");
DEFINE_bool(fcustom_kernel, true, "Register custom kernels during kernel selection.");
DECLARE_bool(fantares_mode);
DECLARE_string(fproduct_name);

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    ProfilingBasedKernelSelector::profiling_best(shared_ptr<GNode> gnode,
                                                 NNFusion_DeviceType devtype,
                                                 IProfilingRuntime::Pointer runtime)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), devtype, element::f32);

    // Skip since only one candidate or constant
    if (kernel_regs.size() == 1 || gnode->is_constant())
        return std::make_pair(devtype, nullptr);

    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    bool has_valid_kernel = false;
    NNFUSION_LOG(INFO) << "Start profiling...";
    auto comparef = [](const ProfilingContext::Pointer& a, const ProfilingContext::Pointer& b) {
        return a->result.get_device_avg() > b->result.get_device_avg();
    };
    priority_queue<ProfilingContext::Pointer, vector<ProfilingContext::Pointer>, decltype(comparef)>
        prof_res(comparef);
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            nnfusion::profiler::Profiler prof(runtime, pctx);

            if (!prof.execute())
                NNFUSION_LOG(INFO) << "Kernel Failed.";
            else
            {
                NNFUSION_LOG(INFO) << "Kernel Emitter#" << prof_res.size()
                                   << " time cost(ms):" << pctx->result.get_device_avg();
                prof_res.push(pctx);
            }
        }
    }

    while (!prof_res.empty())
    {
        auto best = prof_res.top();
        prof_res.pop();
        ///\todo Check if the result is ready.
        if (!best->result.is_ready())
            continue;
        NNFUSION_LOG(INFO) << "Best kernel time cost(ms):" << best->result.get_device_avg();
        return std::make_pair(devtype, move(best->kernel));
    }
    return std::make_pair(devtype, nullptr);
}

bool ProfilingBasedKernelSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool enable_selection = FLAGS_fkernel_selection;
    if (!enable_selection)
        return true;

    // auto dev_name = FLAGS_fdefault_device.c_str();
    // NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

    // Config area
    vector<string> white_list{"Broadcast"};
    //bool all_device = false;
    NNFusion_DeviceType the_device = ROCM_GPU;

    // if (the_device != default_device)
    //     return true;

    // Currently *ONLY* has BroadCast Selection
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if (!(*it)["DeviceType"].is_valid())
        {
            NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                  << it->get_name();
        }
        auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
        NNFUSION_CHECK(n_device_type != UNKNOWN);
        if (n_device_type != the_device)
            continue;
        auto opname = it->get_op_type();
        for (auto& rule : white_list)
            if (opname == rule)
            {
                (*it)["Enable_Kernel_Selection"] = true;
                // if (!all_device)
                //     (*it)["Kernel_Selection_Device"] = the_device;
            }
    }

    for (auto it : nodes)
    {
        if ((*it)["Enable_Kernel_Selection"].is_valid() &&
            (*it)["Enable_Kernel_Selection"].as<bool>())
        {
            auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
            auto ans = profiling_best(it, n_device_type, get_default_runtime(n_device_type));
            if (ans.second != nullptr)
                (*it)["Kernel_Selection_Result"] = ans;
            else
            {
                if (n_device_type == ROCM_GPU)
                {
                    auto ans_cuda = profiling_best(it, CUDA_GPU, get_default_runtime(CUDA_GPU));
                    if (ans_cuda.second != nullptr)
                        (*it)["Kernel_Selection_Result"] = ans_cuda;
                }
            }
        }
    }

    return true;
}

bool DefaultKernelSelector::register_custom_kernel(std::string op, NNFusion_DeviceType devtype)
{
    auto iter = nnfusion::op::get_op_configs().find(op);
    if (iter == nnfusion::op::get_op_configs().end())
        return false;
    if (iter->second.f_kernel_funcs.find(get_device_str(devtype)) ==
        iter->second.f_kernel_funcs.end())
        return false;
    KernelRegistration::Factory f;
    switch (devtype)
    {
    case CUDA_GPU:
    case ROCM_GPU:
        f = [](shared_ptr<kernels::KernelContext> context) -> shared_ptr<kernels::KernelEmitter> {
            return make_shared<kernels::cuda::CustomCudaKernelEmitter>(context);
        };
        break;
    case GENERIC_CPU:
        f = [](shared_ptr<kernels::KernelContext> context) -> shared_ptr<kernels::KernelEmitter> {
            return make_shared<kernels::cpu::CustomCPUKernelEmitter>(context);
        };
        break;
    // case HLSL:
    //     f = [](shared_ptr<kernels::KernelContext> context) -> shared_ptr<kernels::KernelEmitter> {
    //         return make_shared<kernels::cuda::CustomCudaKernelEmitter>(context);
    //     };
    //     break;
    default: NNFUSION_LOG(ERROR) << "Unrecognized device type: " << devtype; return false;
    }

    kernels::KernelRegistrar kernel_registrar(op,
                                              kernels::Name(op)
                                                  .Device(devtype)
                                                  .TypeConstraint(element::f32)
                                                  .Tag("custom")
                                                  .Priority(10)
                                                  .KernelFactory(f)
                                                  .Build());
    return true;
}

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first(shared_ptr<GNode> gnode, NNFusion_DeviceType devtype)
{
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    register_custom_kernel(gnode->get_op_type(), devtype);
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs;
    kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
        gnode->get_op_type(), devtype, element::f32);

    if (devtype == ROCM_GPU)
    {
        for (auto it : KernelRegistry::Global()->FindKernelRegistrations(
                 gnode->get_op_type(), CUDA_GPU, element::f32))
            kernel_regs.push_back(it);
    }

    KernelEmitter::Pointer antares_kernel;
    for (auto kernel_reg : kernel_regs)
    {
        if (kernel_reg->m_tag == "antares")
        {
            antares_kernel = kernel_reg->m_factory(ctx);
            break;
        }
    }

    std::sort(kernel_regs.begin(),
              kernel_regs.end(),
              [&](const shared_ptr<const KernelRegistration>& x,
                  const shared_ptr<const KernelRegistration>& y) {
                  size_t x_prio, y_prio;
                  // If antares_mode is turned on, the priority of antares kernel is always the highest(9),
                  // and we do not modify the antares kernel priority. Otherwise, the priority of tuned
                  // antares kernel is the highest, while the priority of untuned antares kernel is just
                  // higher than reference kernel(0). Hence, we set its priority to 1. For now, the priority
                  // of other kernels is 2~8.
                  if (!FLAGS_fantares_mode && x->m_tag == "antares" && !antares_kernel->is_tuned())
                  {
                      x_prio = 1;
                  }
                  else
                  {
                      x_prio = x->m_priority;
                  }

                  if (!FLAGS_fantares_mode && y->m_tag == "antares" && !antares_kernel->is_tuned())
                  {
                      y_prio = 1;
                  }
                  else
                  {
                      y_prio = y->m_priority;
                  }
                  // the kernel device type may be different with gnode device type,
                  // e.g., ROCM gnode may use CUDA kernel. To ensure kernel of same
                  // device type have higher prioprity, we add 1 to their priority
                  // before comparing.
                  if (devtype == x->m_device_type)
                      x_prio += 1;
                  if (devtype == y->m_device_type)
                      y_prio += 1;

                  if (x_prio != y_prio)
                      return x_prio > y_prio;

                  auto x_type = x->m_factory(ctx)->get_kernel_type();
                  auto y_type = y->m_factory(ctx)->get_kernel_type();
                  if (x_type != y_type)
                      return x_type < y_type;

                  return false;
              });

    for (auto kernel_reg : kernel_regs)
    {
        KernelEmitter::Pointer kernel;
        if (kernel_reg->m_tag == "antares")
            kernel = antares_kernel;
        else
            kernel = kernel_reg->m_factory(ctx);

        // constant kernel emitter will write file to save weights, skip to do it when codegen.
        if (gnode->is_constant() || kernel->get_or_emit_source())
        {
            // if(kernel->get_or_emit_source() != nullptr)
            //    NNFUSION_LOG(NNFUSION_WARNING) << "Valid kernel found:" << gnode->get_name();
            return std::make_pair(devtype, kernel);
        }
    }

    NNFUSION_LOG(ERROR) << "No valid kernel found:" << gnode->get_name()
                        << "(op type: " << gnode->get_op_type()
                        << ", dev type: " << nnfusion::get_device_str(devtype)
                        << ", dtype: " << gnode->get_element_type() << ")";
    return std::make_pair(devtype, nullptr);
}

bool DefaultKernelSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    // std::vector<std::shared_ptr<GNode>> nodes = graph->get_ordered_ops();
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if (!(*it)["Kernel_Selection_Result"].is_valid())
        {
            if (!(*it)["DeviceType"].is_valid())
            {
                NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                      << it->get_name();
            }
            auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
            NNFUSION_CHECK(n_device_type != UNKNOWN);

            auto ans = pick_first(it, n_device_type);
            if (ans.second != nullptr)
                (*it)["Kernel_Selection_Result"] = ans;
        }
    }

    return true;
}

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    FetchBasedSelector::fetch_inventory(shared_ptr<cache::KernelCacheManager> cache_manager,
                                        shared_ptr<GNode> gnode,
                                        NNFusion_DeviceType devtype)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), devtype, element::f32);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    std::vector<nlohmann::json> functions;

    std::string identifier = ctx->generate_identifier();
    // Todo: platform interface to be coordinated with nnfusion devtype
    const std::vector<std::string> SUPPORT_PLATFORM = {"CUDA_GPU"};

    if (identifier != "" &&
        find(SUPPORT_PLATFORM.begin(), SUPPORT_PLATFORM.end(), get_device_str(devtype)) !=
            SUPPORT_PLATFORM.end())
    {
        // fetch all available kernel entries from kernel cache DB
        auto fetched = cache_manager->fetch_all(identifier, get_device_str(devtype));

        // emit External kernels
        {
            for (auto kernel_entry : fetched)
            {
                if (kernel_entry->source == "External")
                {
                    NNFUSION_CHECK(kernel_entry->tags.find("BlockCudaEmitter") !=
                                   kernel_entry->tags.end());
                    auto kernel =
                        std::make_shared<kernels::cuda::CacheBlockCudaEmitter>(ctx, kernel_entry);
                    if (kernel->get_or_emit_source())
                    {
                        return std::make_pair(devtype, kernel);
                    }
                }
            }
        }

        // emit Antares kernels: select the fastest tuned kernel
        {
            nnfusion::cache::KernelEntry_p kernel_entry = nullptr;
            double kernel_time = 1000000000;
            for (auto fetch_entry : fetched)
            {
                if (fetch_entry->source == "Antares")
                {
                    if (fetch_entry->miscs["antares"]["device_name"] != FLAGS_fproduct_name)
                    {
                        continue;
                    }
                    if (fetch_entry->miscs["antares"]["planned_steps"] > 0)
                    {
                        if (kernel_entry == nullptr ||
                            fetch_entry->miscs["antares"]["time"] < kernel_time)
                        {
                            kernel_entry = fetch_entry;
                            kernel_time = fetch_entry->miscs["antares"]["time"];
                        }
                    }
                }
            }
            if (kernel_entry != nullptr)
            {
                if (kernel_entry->tags.find("BlockCudaEmitter") != kernel_entry->tags.end())
                {
                    auto kernel =
                        std::make_shared<kernels::cuda::CacheBlockCudaEmitter>(ctx, kernel_entry);
                    if (kernel->get_or_emit_source())
                    {
                        return std::make_pair(devtype, kernel);
                    }
                }
                else
                {
                    NNFUSION_CHECK(kernel_entry->tags.find("CudaEmitter") !=
                                   kernel_entry->tags.end());
                    auto kernel =
                        std::make_shared<kernels::cuda::CacheCudaEmitter>(ctx, kernel_entry);
                    if (kernel->get_or_emit_source())
                    {
                        return std::make_pair(devtype, kernel);
                    }
                }
            }
        }
    }
    return std::make_pair(devtype, nullptr);
}

bool FetchBasedSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    auto cache_manager = std::make_shared<cache::KernelCacheManager>();
    if (!cache_manager->is_valid())
    {
        NNFUSION_LOG(INFO) << "No valid kernel cache, FetchBasedSelector will be skipped";
        return true;
    }
    // auto dev_name = FLAGS_fdefault_device.c_str();
    // NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if (!(*it)["Kernel_Selection_Result"].is_valid())
        {
            if (!(*it)["DeviceType"].is_valid())
            {
                NNFUSION_CHECK_FAIL() << "GNode DeviceType should be assigned before this pass："
                                      << it->get_name();
            }
            auto n_device_type = (*it)["DeviceType"].as<NNFusion_DeviceType>();
            NNFUSION_CHECK(n_device_type != UNKNOWN);
            auto ans = fetch_inventory(cache_manager, it, n_device_type);

            if (ans.second != nullptr)
                (*it)["Kernel_Selection_Result"] = ans;
        }
    }

    return true;
}