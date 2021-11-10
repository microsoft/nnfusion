// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "runtime_const_folding_pass.hpp"

DEFINE_string(fconst_folding_backend,
              "",
              "Choose which backend will be used in Constant folding pass. Disable when not set.");

using namespace nnfusion::pass::graph;

std::shared_ptr<GNode> RuntimeConstantFoldingPass::runtime_const_folding_node(
    std::shared_ptr<Graph>& graph,
    std::set<std::shared_ptr<GNode>>& blocklist_nodes,
    std::shared_ptr<GNode>& node)
{
    NNFUSION_LOG(INFO) << ">> Found constant downstream node: " << node->get_name()
                       << ", Op Type = " << node->get_op_type();

    bool const_infer_success = false;
    std::vector<std::vector<char>> raw_inputs(node->get_input_size()), raw_outputs;

    // Prepare constant inputs from upstream_nodes
    std::set<std::shared_ptr<GNode>> upstream_nodes;
    for (auto& input : node->get_in_edges())
    {
        if (input->is_control_edge())
            continue;
        auto const_node = input->get_src();
        NNFUSION_LOG(INFO) << "  Input of constant downstream node: " << const_node->get_name()
                           << ", Op Type = " << const_node->get_op_type() << "/"
                           << const_node->get_op_type();

        NNFUSION_CHECK(input->get_dst() == node);
        NNFUSION_CHECK(const_node->is_constant());
        upstream_nodes.insert(const_node);

        auto p_const = std::dynamic_pointer_cast<op::Constant>(const_node->get_op_ptr());
        NNFUSION_CHECK(p_const != nullptr);
        const void* ptr = p_const->get_data_ptr();
        size_t length = p_const->get_data_size();
        NNFUSION_LOG(INFO) << "  With Constant Input Node: " << p_const->get_name()
                           << ", Memory Length = " << length;

        size_t input_index = input->get_dst_input();
        raw_inputs[input_index].resize(length);
        memcpy(raw_inputs[input_index].data(), ptr, length);
    }

    // Prepare runtime backend
    nnfusion::profiler::IProfilingRuntime::Pointer runtime = nullptr;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs;

    if (backend == "ROCm")
    {
        runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
        NNFUSION_CHECK(runtime->check_env());
        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
            node->get_op_type(), ROCM_GPU, element::f32);
        if (kernel_regs.size() == 0)
            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                node->get_op_type(), CUDA_GPU, element::f32);
    }
    else if (backend == "CUDA")
    {
        runtime = nnfusion::profiler::CudaDefaultRuntime::Runtime();
        NNFUSION_CHECK(runtime->check_env());
        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
            node->get_op_type(), CUDA_GPU, element::f32);
    }
    else if (backend == "CPU")
    {
        runtime = nnfusion::profiler::ReferenceRuntime::Runtime();
        NNFUSION_CHECK(runtime->check_env());
        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
            node->get_op_type(), GENERIC_CPU, element::f32);
    }
    else
    {
        NNFUSION_CHECK_FAIL() << "Cannot Recognize Backend Type: " << backend;
    }

    // Runtime node output inference
    shared_ptr<KernelContext> ctx(new KernelContext(node));
    for (auto& kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (!kernel->get_or_emit_source())
            continue;
        if (!this->fast_debug)
        {
            nnfusion::profiler::ProfilingContext::Pointer pctx =
                make_shared<nnfusion::profiler::ProfilingContext>(kernel, false);

            nnfusion::profiler::Profiler prof(runtime, pctx);
            if (!prof.mixed_type_execute(raw_inputs, raw_outputs))
                continue;
        }
        else
        {
            raw_outputs.resize(node->get_output_size());
            for (int i = 0; i < raw_outputs.size(); ++i)
            {
                auto& shape = node->get_output_shape(i);
                auto size = node->get_output_element_type(i).size();
                for (auto& it : shape)
                    size *= it;
                raw_outputs[i].resize(size);
                memset(raw_outputs[i].data(), 0, raw_outputs[i].size());
            }
        }
        NNFUSION_LOG(INFO) << "  For node `" << node->get_name()
                           << "`: get runtime output results of size " << raw_outputs.size();
        const_infer_success = true;
        break;
    }
    if (!const_infer_success)
    {
        NNFUSION_LOG(INFO) << "  For node `" << node->get_name()
                           << "`: Cannot infer outputs, going to blacklist this node.";
        blocklist_nodes.insert(node);
        return nullptr;
    }

    // Only support single output; Multi-outputs lacks output-index properties in GNode.
    NNFUSION_CHECK(raw_outputs.size() == 1);
#if 0 // For Debug only
                    NNFUSION_LOG(INFO) << "inputs = ";
                    for (int i = 0; i < std::min(raw_inputs[0].size() / 4, 10LU); ++i)
                        NNFUSION_LOG(INFO) << (float*)raw_inputs[0].data())[i];
                    puts("..");

                    NNFUSION_LOG(INFO) << "outputs = ";
                    for (int i = 0; i < std::min(raw_outputs[0].size() / 4, 10LU); ++i)
                        NNFUSION_LOG(INFO) << (float*)raw_outputs[0].data())[i];
                    puts("..");
#endif
    // Ensure output layout is as expected, replace node with new_constant in place
    std::shared_ptr<nnfusion::graph::GNode> result = nullptr;
    NNFUSION_CHECK(raw_outputs.size() == node->get_output_size());
    for (int i = 0; i < node->get_output_size(); ++i)
    {
        auto& shape = node->get_output_shape(i);
        auto& dtype = node->get_output_element_type(i);
        size_t memory = dtype.size();
        for (auto& it : shape)
            memory *= it;
        NNFUSION_CHECK(memory == raw_outputs[i].size());

        // 1. create new constant node
        std::shared_ptr<op::Constant> new_constant_op;
        new_constant_op = std::make_shared<op::Constant>(dtype, shape, raw_outputs[i].data());
        auto new_constant_gnode =
            std::make_shared<nnfusion::graph::GNode>(new_constant_op, GNodeVector());

        graph->replace_node(node, new_constant_gnode, false);
        result = new_constant_gnode;

        // remove upstream nodes with 0 out-degree
        for (auto& node : upstream_nodes)
        {
            if (node->get_out_edges().size() == 0)
            {
                graph->remove_node(node);
            }
        }

        NNFUSION_LOG(INFO) << "  Finish folding node: name = " << node->get_unique_name() << "/"
                           << node->get_name() << ", type = " << node->get_op_type();
        NNFUSION_LOG(INFO) << "";
    }
    return result;
}

bool RuntimeConstantFoldingPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    int at = FLAGS_fconst_folding_backend.find(":DEBUG");
    if (at >= 0)
    {
        this->backend = FLAGS_fconst_folding_backend.substr(0, at);
        this->fast_debug = true;
    }
    else
    {
        this->backend = FLAGS_fconst_folding_backend;
        this->fast_debug = false;
    }

    if (this->backend == "")
        return true;

    static bool has_warning = false;
    if (!has_warning)
    {
        has_warning = true;
    }

    NNFUSION_LOG(INFO) << "Runtime Constant Folding Pass starts up for Graph: "
                       << graph->get_name();

    // Folding output nodes results in kernel_emitter crashes
    std::set<std::shared_ptr<GNode>> blocklist_nodes = {};
    for (auto& node : graph->get_outputs())
        blocklist_nodes.insert(node);

    if (pool_ptr == nullptr)
    {
        pool_ptr = std::make_shared<thread_pool>();
    }
    run_on_graph_parallel(graph, blocklist_nodes);

    NNFUSION_LOG(INFO) << "";
    NNFUSION_LOG(INFO) << ">> Runtime Constant Folding Pass ends for Graph: " << graph->get_name();
    NNFUSION_LOG(INFO) << "";
    return true;
}

RuntimeConstantFoldingPass::thread_pool::thread_pool()
    : stopped{false}
{
    total_thread_num = std::thread::hardware_concurrency();
    idl_thread_num = total_thread_num;
    for (int size = 0; size < total_thread_num; ++size)
    {
        pool.emplace_back([this] {
            while (!this->stopped)
            {
                Task task;
                {
                    std::unique_lock<std::mutex> lock{this->m_lock};
                    this->cv_task.wait(
                        lock, [this] { return this->stopped.load() || !this->tasks.empty(); });
                    if (this->stopped && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                idl_thread_num--;
                task();
                idl_thread_num++;
            }
        });
    }
}

RuntimeConstantFoldingPass::thread_pool::~thread_pool()
{
    stopped.store(true);
    cv_task.notify_all();
    for (std::thread& thread : pool)
    {
        if (thread.joinable())
            thread.join();
    }
}

void RuntimeConstantFoldingPass::thread_pool::commit(Task task)
{
    if (stopped.load())
    {
        NNFUSION_LOG(ERROR) << "Runtime Constant Folding Pass Commit on ThreadPool is stopped.";
        return;
    }
    {
        std::lock_guard<std::mutex> lock{m_lock};
        tasks.emplace([this, task]() {
            task();
            this->cv_task_done.notify_one();
        });
    }
    cv_task.notify_one();
}

bool RuntimeConstantFoldingPass::thread_pool::is_free()
{
    std::unique_lock<std::mutex> lock{m_lock};
    return tasks.empty() && idl_thread_num == total_thread_num;
}

void RuntimeConstantFoldingPass::thread_pool::wait_for_all()
{
    std::unique_lock<std::mutex> lock{m_lock_done};
    cv_task_done.wait(lock, [this]() { return this->is_free(); });
}

void RuntimeConstantFoldingPass::runtime_const_folding_task(
    std::shared_ptr<Graph>& graph,
    std::set<std::shared_ptr<GNode>>& blocklist_nodes,
    std::shared_ptr<GNode> node,
    std::map<std::shared_ptr<GNode>, int>& in_degree,
    std::mutex& in_degree_lock)
{
    if (!node->is_constant())
    {
        node = runtime_const_folding_node(graph, blocklist_nodes, node);
        if (node == nullptr)
        {
            return;
        }
    }

    for (auto& output : node->get_out_edges())
    {
        if (output->is_control_edge())
        {
            continue;
        }
        auto target = output->get_dst();
        NNFUSION_CHECK(output->get_src() == node);
        {
            std::lock_guard<std::mutex> lock{in_degree_lock};
            auto it = in_degree.find(target);
            if (it != in_degree.end() && it->second > 0)
            {
                it->second -= 1;
                if (it->second == 0)
                {
                    pool_ptr->commit(
                        [this, &graph, &blocklist_nodes, &target, &in_degree, &in_degree_lock]() {
                            this->runtime_const_folding_task(
                                graph, blocklist_nodes, target, in_degree, in_degree_lock);
                        });
                }
            }
            else
            {
                NNFUSION_LOG(ERROR) << "Update target node's indegree error.";
            }
        }
    }
}

bool RuntimeConstantFoldingPass::run_on_graph_parallel(
    std::shared_ptr<Graph>& graph, std::set<std::shared_ptr<GNode>>& blocklist_nodes)
{
    std::map<std::shared_ptr<GNode>, int> in_degree;
    std::mutex in_degree_lock;

    int folding_cnt = 0;
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    std::set<std::shared_ptr<GNode>> const_nodes = {};
    std::set<std::shared_ptr<GNode>> down_streams = {};

    for (auto& it : nodes)
    {
        for (auto& output : it->get_out_edges())
        {
            if (output->is_control_edge())
            {
                continue;
            }
            auto target = output->get_dst();
            in_degree[target] += 1;
        }
    }
    for (auto& it : nodes)
    {
        if (in_degree[it] == 0)
        {
            pool_ptr->commit([this, &graph, &blocklist_nodes, &it, &in_degree, &in_degree_lock]() {
                this->runtime_const_folding_task(
                    graph, blocklist_nodes, it, in_degree, in_degree_lock);
            });
        }
    }
    pool_ptr->wait_for_all();
}
