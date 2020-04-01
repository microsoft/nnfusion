// Microsoft (c) 2019, NNFusion Team
#include "kernel_selection.hpp"

#include <queue>
#include <utility>

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

// Register_Tag(Enable_Kernel_Selection, bool);
// Register_Tag(Kernel_Selection_Device, DeviceType);
// Register_Tag(Kernel_Selection_Result, vector<pair<DeviceType, KernelEmitter>>);

DEFINE_bool(fkernel_selection, true, "Select kernel before codegen.");
DEFINE_bool(fkernel_tunning, false, "Tunning and choose best kernel when do kernel selection.");

pair<DeviceType, kernels::KernelEmitter::Pointer> ProfilingBasedKernelSelector::profiling_best(
    shared_ptr<GNode> gnode, DeviceType devtype, IProfilingRuntime::Pointer runtime)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);

    // Skip since only one candidate or constant
    if (kernel_regs.size() == 1 || gnode->is_constant())
        return std::make_pair(devtype, nullptr);

    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    bool has_valid_kernel = false;
    LOG(INFO) << "Start profiling...";
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
                LOG(INFO) << "Kernel Failed.";
            else
            {
                LOG(INFO) << "Kernel Emitter#" << prof_res.size()
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
        LOG(INFO) << "Best kernel time cost(ms):" << best->result.get_device_avg();
        return std::make_pair(devtype, move(best->kernel));
    }
    return std::make_pair(devtype, nullptr);
}

bool ProfilingBasedKernelSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool enable_tuning = FLAGS_fkernel_tunning;
    if (!enable_tuning)
        return true;

    // Config area
    vector<string> white_list{"Broadcast"};
    bool all_device = false;
    DeviceType the_device = ROCM_GPU;

    // Currently *ONLY* has BroadCast Selection
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        auto opname = it->get_op_type();
        for (auto& rule : white_list)
            if (opname == rule)
            {
                (*it)["Enable_Kernel_Selection"] = true;
                if (!all_device)
                    (*it)["Kernel_Selection_Device"] = the_device;
            }
    }

    for (auto it : nodes)
    {
        if ((*it)["Enable_Kernel_Selection"].is_valid() &&
            (*it)["Enable_Kernel_Selection"].as<bool>())
        {
            (*it)["Kernel_Selection_Result"] = vector<pair<DeviceType, KernelEmitter::Pointer>>();
            auto& res = (*it)["Kernel_Selection_Result"]
                            .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();

            vector<DeviceType> dev_type{CUDA_GPU, ROCM_GPU, GENERIC_CPU};
            for (auto t : dev_type)
            {
                if ((*it)["Kernel_Selection_Device"].is_valid() &&
                    (*it)["Kernel_Selection_Device"].as<DeviceType>() != t)
                    continue;

                auto ans = profiling_best(it, t, get_default_runtime(t));

                if (ans.second != nullptr)
                    res.push_back(ans);
            }
        }
    }

    return true;
}

pair<DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first(shared_ptr<GNode> gnode, DeviceType devtype)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), devtype, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        // constant kernel emitter will write file to save weights, skip to do it when codegen.
        if (gnode->is_constant() || kernel->get_or_emit_source())
        {
            // if(kernel->get_or_emit_source() != nullptr)
            //    LOG(WARNING) << "Valid kernel found:" << gnode->get_name();
            return std::make_pair(devtype, kernel);
        }
    }
    LOG(ERROR) << "No valid kernel found:" << gnode->get_name();
    return std::make_pair(devtype, nullptr);
}

pair<DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first_rocm(shared_ptr<GNode> gnode)
{
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));
    auto kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(gnode->get_op_type(), ROCM_GPU, DT_FLOAT);
    if (!kernel_regs.size())
        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), CUDA_GPU, DT_FLOAT);
    else
    {
        auto priority = [](const std::string& tag) -> int {
            static char sym_prio[] = "PRIORITY_";
            int at = tag.find(sym_prio);
            return (at != 0) ? 0 : atoi(tag.substr(sizeof(sym_prio) - 1).c_str());
        };

        std::sort(kernel_regs.begin(),
                  kernel_regs.end(),
                  [&](const shared_ptr<const KernelRegistration>& x,
                      const shared_ptr<const KernelRegistration>& y) {
                      auto x_prio = priority(x->m_tag), y_prio = priority(y->m_tag);
                      if (x_prio != y_prio)
                          return x_prio > y_prio;

                      auto x_type = x->m_factory(ctx)->get_kernel_type();
                      auto y_type = y->m_factory(ctx)->get_kernel_type();
                      if (x_type != y_type)
                          return x_type < y_type;

                      return false;
                  });
    }

    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (gnode->is_constant() || kernel->get_or_emit_source())
        {
            return std::make_pair(ROCM_GPU, kernel);
        }
    }
    LOG(ERROR) << "No valid kernel found:" << gnode->get_name();
    return std::make_pair(ROCM_GPU, nullptr);
}

bool DefaultKernelSelector::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        if (!(*it)["Kernel_Selection_Result"].is_valid())
            (*it)["Kernel_Selection_Result"] = vector<pair<DeviceType, KernelEmitter::Pointer>>();
        auto& res =
            (*it)["Kernel_Selection_Result"].as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();

        vector<DeviceType> dev_type{CUDA_GPU, ROCM_GPU, GENERIC_CPU};
        for (auto t : dev_type)
        {
            if ((*it)["Kernel_Selection_Device"].is_valid() &&
                (*it)["Kernel_Selection_Device"].as<DeviceType>() != t)
                continue;

            bool selected = false;
            for (auto& p : res)
            {
                if (p.first == t)
                {
                    selected = true;
                    break;
                }
            }
            if (selected)
                continue;

            if (t == ROCM_GPU)
            {
                auto ans = pick_first_rocm(it);
                res.push_back(ans);
            }
            else
            {
                auto ans = pick_first(it, t);
                res.push_back(ans);
            }
        }
    }

    return true;
}
