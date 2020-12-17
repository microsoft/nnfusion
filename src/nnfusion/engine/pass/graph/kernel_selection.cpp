
// Microsoft (c) 2019, NNFusion Team
#include "kernel_selection.hpp"
#include <queue>
#include <utility>
#include "nnfusion/core/kernels/cpu/cpu_kernel_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

// Register_Tag(Enable_Kernel_Selection, bool);
// Register_Tag(Kernel_Selection_Device, NNFusion_DeviceType);
// Register_Tag(Kernel_Selection_Result, vector<pair<NNFusion_DeviceType, KernelEmitter>>);

DEFINE_bool(fkernel_selection, true, "Select kernel before codegen.");
DEFINE_bool(fkernel_tunning, false, "Tunning and choose best kernel when do kernel selection.");
DECLARE_bool(fantares_mode);
DEFINE_bool(fprof_all_non_antares_kernels,
            false,
            "Profiling all non-antares kernels when using AntaresProfilingBasedKernelSelector.");
DEFINE_bool(fantares_prof_kernel_selector, false, "Use AntaresProfilingBasedKernelSelector");

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    ProfilingBasedKernelSelector::profiling_best(shared_ptr<GNode> gnode,
                                                 NNFusion_DeviceType devtype,
                                                 IProfilingRuntime::Pointer runtime)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);

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
    bool enable_tuning = FLAGS_fkernel_tunning;
    if (!enable_tuning)
        return true;
    if (FLAGS_fantares_prof_kernel_selector)
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

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first(shared_ptr<GNode> gnode, NNFusion_DeviceType devtype)
{
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);

    if (devtype == ROCM_GPU)
    {
        for (auto it : KernelRegistry::Global()->FindKernelRegistrations(
                 gnode->get_op_type(), CUDA_GPU, DT_FLOAT))
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
                        << ", dev type: " << nnfusion::get_device_str(devtype) << ")";
    return std::make_pair(devtype, nullptr);
}

bool DefaultKernelSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fantares_prof_kernel_selector)
        return true;
    register_antares_kernel();
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

std::string nnfusion::pass::graph::generate_identifier(const shared_ptr<KernelContext>& ctx)
{
    std::string op_type = ctx->gnode->get_op_type();

    // identifier of pattern substitution kernel was generated before
    if (op_type == "Matched_Pattern")
        return (*ctx->gnode)["identifier"].as<std::string>();

    // Todo: more spec to be added
    std::string identifier("");

    // operator type as identifier
    identifier += op_type;

    // shapes of input tensors as identifier
    for (int i = 0; i < ctx->inputs.size(); ++i)
    {
        auto& shape = ctx->inputs[i]->get_shape();
        for (int j = 0; j < shape.size(); ++j)
            identifier += to_string(shape[j]);
    }

    // shapes of output tensors as identifier
    for (int i = 0; i < ctx->outputs.size(); ++i)
    {
        auto& shape = ctx->outputs[i]->get_shape();
        for (int j = 0; j < shape.size(); ++j)
            identifier += to_string(shape[j]);
    }

    // data types of input tensors as identifier
    for (int i = 0; i < ctx->dtypes.size(); ++i)
    {
        identifier += ctx->dtypes[i];
    }

    if (op_type == "Convolution")
    {
        auto conv = std::dynamic_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(conv);
        std::stringstream str;
        str << conv->get_window_movement_strides();
        str << conv->get_window_dilation_strides();
        str << conv->get_padding_below();
        identifier += str.str();
    }
    else if (op_type == "AvgPool")
    {
        auto avgpool = std::dynamic_pointer_cast<op::AvgPool>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(avgpool);
        std::stringstream str;
        str << avgpool->get_window_shape();
        str << avgpool->get_window_movement_strides();
        str << avgpool->get_padding_below();
        identifier += str.str();
    }
    else if (op_type == "MaxPool")
    {
        auto maxpool = std::dynamic_pointer_cast<op::MaxPool>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(maxpool);
        std::stringstream str;
        str << maxpool->get_window_shape();
        str << maxpool->get_window_movement_strides();
        str << maxpool->get_padding_below();
        identifier += str.str();
    }
    else if (op_type == "Dot")
    {
        ///\todo encode dot attrs, stay the same with db importor
        // auto dot = std::dynamic_pointer_cast<op::Dot>(ctx->gnode->get_op_ptr());
        // NNFUSION_CHECK_NOT_NULLPTR(dot);
        // std::stringstream str;
        // str << dot->get_transpose_A();
        // str << dot->get_transpose_B();
        // ///\todo: need to encode dot reduction_axes_count?
        // identifier += str.str();
    }

    return identifier;
}

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    FetchBasedSelector::fetch_inventory(shared_ptr<cache::KernelCacheManager> cache_manager,
                                        shared_ptr<GNode> gnode,
                                        NNFusion_DeviceType devtype)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    std::vector<std::string> functions;

    std::string identifier = generate_identifier(ctx);
    // Todo: platform interface to be coordinated with nnfusion devtype
    std::vector<std::string> platform = {"CUDA"};

    if (identifier != "")
    {
        // Todo: more tags and policy to be added
        std::set<std::string> tags = {"fast"};
        auto fetched_kernel = cache_manager->fetch_with_tags(identifier, platform.front(), tags);
        if (fetched_kernel.function != "")
        {
            functions.push_back(fetched_kernel.function);
        }
    }
    if (functions.size() != 0)
    {
        // Todo: more policy to be added
        for (auto func : functions)
        {
            auto kernel = std::make_shared<kernels::cuda::CacheBlockCudaKernel>(ctx, func);
            if (kernel->get_or_emit_source())
            {
                return std::make_pair(devtype, kernel);
            }
        }
    }
    else
    {
        return std::make_pair(devtype, nullptr);
    }
}

bool FetchBasedSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fantares_prof_kernel_selector)
        return true;
    auto cache_manager = std::make_shared<cache::KernelCacheManager>();
    if (!cache_manager->is_valid())
    {
        NNFUSION_LOG(INFO) << "No valid kernel cache, default selector will be used";
        auto selector = DefaultKernelSelector();
        return selector.run_on_graph(graph);
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

pair<NNFusion_DeviceType, kernels::KernelEmitter::Pointer>
    AntaresProfilingBasedKernelSelector::pick_best(shared_ptr<GNode> gnode,
                                                   NNFusion_DeviceType devtype)
{
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);

    if (devtype == ROCM_GPU)
    {
        for (auto it : KernelRegistry::Global()->FindKernelRegistrations(
                 gnode->get_op_type(), CUDA_GPU, DT_FLOAT))
            kernel_regs.push_back(it);
    }

    std::sort(kernel_regs.begin(),
              kernel_regs.end(),
              [&](const shared_ptr<const KernelRegistration>& x,
                  const shared_ptr<const KernelRegistration>& y) {
                  size_t x_prio, y_prio;
                  x_prio = x->m_priority;
                  y_prio = y->m_priority;

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

    std::vector<KernelEmitter::Pointer> non_antares_kernels;
    KernelEmitter::Pointer antares_kernel;
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        // constant kernel emitter will write file to save weights, skip to do it when codegen.
        if (gnode->is_constant())
        {
            return std::make_pair(devtype, kernel);
        }
        if (kernel->get_or_emit_source())
        {
            if (kernel_reg->m_tag == "antares")
                antares_kernel = kernel;
            else
            {
                non_antares_kernels.push_back(kernel);
            }
        }
    }

    if (non_antares_kernels.empty())
    {
        if (antares_kernel)
        {
            NNFUSION_LOG(INFO) << "No non-antares kernel found, choose antares kernel: "
                               << gnode->get_name() << "(op type: " << gnode->get_op_type()
                               << ", dev type: " << nnfusion::get_device_str(devtype) << ")";
            return std::make_pair(devtype, antares_kernel);
        }
        else
        {
            NNFUSION_LOG(ERROR) << "No valid kernel found:" << gnode->get_name()
                                << "(op type: " << gnode->get_op_type()
                                << ", dev type: " << nnfusion::get_device_str(devtype) << ")";
            return std::make_pair(devtype, nullptr);
        }
    }

    // Because this kernel selector targets at comparing the performance of antares kernel with that
    // of non-antares kernel, if antares_kernel does not exist, we choose the first non-antares kernel
    // without profiling.
    if (!antares_kernel)
    {
        NNFUSION_LOG(INFO) << "No antares kernel found, choose non-antares kernel: "
                           << gnode->get_name() << "(op type: " << gnode->get_op_type()
                           << ", dev type: " << nnfusion::get_device_str(devtype) << ")";
        return std::make_pair(devtype, non_antares_kernels[0]);
    }

    static unordered_map<string, int> ir_kernel_idx;
    std::string ir = nnfusion::op::get_translation(gnode);

    if (!ir.empty() && ir_kernel_idx.find(ir) != ir_kernel_idx.end())
    {
        NNFUSION_LOG(INFO) << "Using cache, omit profiling.";
        int idx = ir_kernel_idx[ir];
        // -1 represents antares kernel
        if (idx == -1)
            return std::make_pair(devtype, antares_kernel);
        else
            return std::make_pair(devtype, non_antares_kernels[idx]);
    }

    NNFUSION_LOG(INFO) << "Start profiling...";

    auto comparef = [](const std::pair<ProfilingContext::Pointer, int>& a,
                       const std::pair<ProfilingContext::Pointer, int>& b) {
        return a.first->result.get_device_avg() > b.first->result.get_device_avg();
    };

    priority_queue<std::pair<ProfilingContext::Pointer, int>,
                   vector<std::pair<ProfilingContext::Pointer, int>>,
                   decltype(comparef)>
        prof_res(comparef);

    auto prof = [&prof_res](KernelEmitter::Pointer kernel, NNFusion_DeviceType devtype, int idx) {

        auto profiling_kernel = [](KernelEmitter::Pointer kernel,
                                   IProfilingRuntime::Pointer runtime)
            -> std::shared_ptr<nnfusion::profiler::ProfilingContext> {
            auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            nnfusion::profiler::Profiler prof(runtime, pctx);

            if (prof.execute())
            {
                double kernel_time = pctx->result.get_device_avg();
                NNFUSION_LOG(INFO) << "Profiling kernel: " << kernel->get_function_name()
                                   << ", kernel time(ms):" << kernel_time;
                return pctx;
            }
            else
            {
                NNFUSION_LOG(INFO) << "Kernel Failed.";
                return nullptr;
            }
        };

        std::shared_ptr<nnfusion::profiler::ProfilingContext> result;
        if (devtype == CUDA_GPU)
        {
            result = profiling_kernel(kernel, CudaDefaultRuntime::Runtime());
        }
        else if (devtype == GENERIC_CPU)
        {
            result = profiling_kernel(kernel, CPUDefaultRuntime::Runtime());
        }
        else
        {
            result = profiling_kernel(kernel, get_default_runtime(devtype));
            if (!result && devtype == ROCM_GPU)
            {
                result = profiling_kernel(kernel, get_default_runtime(CUDA_GPU));
            }
        }

        if (result)
            prof_res.push(std::make_pair(result, idx));
    };

    prof(antares_kernel, devtype, -1);
    if (FLAGS_fprof_all_non_antares_kernels)
    {
        for (int i = 0; i < non_antares_kernels.size(); i++)
            prof(non_antares_kernels[i], devtype, i);
    }
    else
    {
        prof(non_antares_kernels[0], devtype, 0);
    }

    if (!prof_res.empty())
    {
        auto best = prof_res.top();
        if (!ir.empty())
            ir_kernel_idx[ir] = best.second;
        bool is_antares_kernel = (best.second == -1);
        NNFUSION_LOG(INFO) << "Best kernel time cost(ms):" << best.first->result.get_device_avg()
                           << " (is antares kernel: " << is_antares_kernel << ")";
        ;
        return std::make_pair(devtype, move(best.first->kernel));
    }
    else
    {
        // NNFUSION_LOG(ERROR) << "No valid kernel found:" << gnode->get_name()
        //                         << "(op type: " << gnode->get_op_type()
        //                         << ", dev type: " << nnfusion::get_device_str(devtype) << ")";
        // return std::make_pair(devtype, nullptr);
        NNFUSION_LOG(INFO) << "Profiling failed. Choose the first non-antares kernel.";
        if (!ir.empty())
            ir_kernel_idx[ir] = 0;
        return std::make_pair(devtype, non_antares_kernels[0]);
    }
}

bool AntaresProfilingBasedKernelSelector::run_on_graph(
    std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fantares_prof_kernel_selector)
        return true;
    register_antares_kernel();
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

            auto ans = pick_best(it, n_device_type);
            if (ans.second != nullptr)
                (*it)["Kernel_Selection_Result"] = ans;
        }
    }

    return true;
}

bool nnfusion::pass::graph::register_antares_kernel()
{
    for (auto pair : nnfusion::op::get_op_configs())
    {
        std::string op_name = pair.first;
        std::vector<NNFusion_DeviceType> devs{CUDA_GPU, GENERIC_CPU, HLSL};

        KernelRegistrar kernel_registrar_cuda(
            op_name,
            Name(op_name)
                .Device(CUDA_GPU)
                .TypeConstraint(DT_FLOAT)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {
                    return make_shared<cuda::AntaresCudaKernelEmitter>(context);
                })
                .Build());
        KernelRegistrar kernel_registrar_cpu(
            op_name,
            Name(op_name)
                .Device(GENERIC_CPU)
                .TypeConstraint(DT_FLOAT)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {
                    return make_shared<cpu::AntaresCpuKernelEmitter>(context);
                })
                .Build());
        KernelRegistrar kernel_registrar_hlsl(
            op_name,
            Name(op_name)
                .Device(HLSL)
                .TypeConstraint(DT_FLOAT)
                .Tag("antares")
                .Priority(9)
                .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {
                    return make_shared<hlsl::AntaresHLSLKernelEmitter>(context);
                })
                .Build());
    }
    return true;
}