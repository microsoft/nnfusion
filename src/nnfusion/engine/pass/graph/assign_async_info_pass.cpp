// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "assign_async_info_pass.hpp"
#include <queue>
#include "kernel_profiling_pass.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/util/util.hpp"
using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace nnfusion::async;
using namespace nnfusion::kernels::cuda;

DECLARE_bool(fadd_allreduce);
DECLARE_string(fdefault_device);
DECLARE_bool(frt_const_folding);
DEFINE_int32(fnum_stream, 1, "Number of streams. 0 means unlimited stream numbers.");
DECLARE_int32(fnum_non_cpu);
DEFINE_bool(fuse_default_stream, true, "Use default stream.");
DEFINE_string(fcuda_init_stream, "default", "The stream of kernels in cuda_init().");
DECLARE_bool(fenable_kernel_profiling);
DEFINE_string(fstream_assign_policy,
              "naive",
              "Choose stream-assign policy from [naive, kernel_prof_based].");

AssignAsyncInfoPass::AssignAsyncInfoPass()
{
}

bool AssignAsyncInfoPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    auto default_device = nnfusion::get_device_type(dev_name);
    if (default_device == CUDA_GPU || default_device == ROCM_GPU)
    {
        init_assign_async_info(graph);
        gpu_assign_thread_info(graph);
        if (FLAGS_fstream_assign_policy == "kernel_prof_based")
        {
            kernel_prof_based_assign_stream_info(graph);
        }
        else
        {
            naive_assign_stream_info(graph);
            assign_event_info(graph);
        }
    }
    else if (default_device == GENERIC_CPU)
    {
        init_assign_async_info(graph);
        if (FLAGS_fstream_assign_policy == "kernel_prof_based")
        {
            kernel_prof_based_assign_thread_info(graph);
        }
        else
        {
            naive_assign_thread_info(graph);
            assign_event_info(graph);
        }
    }
    else
    {
        assign_default_info(graph);
    }

    NNFUSION_LOG(INFO) << "run async-------------------------------";
    return true;
}

// assign thread for cuda
void AssignAsyncInfoPass::gpu_assign_thread_info(shared_ptr<Graph>& graph)
{
    auto host_async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    int num_gpu = FLAGS_fnum_non_cpu;
    if (num_gpu < 1)
        num_gpu = 1;

    size_t num_async_node = 0;
    static const std::unordered_set<std::string> async_node = {"AllReduce"};

    for (auto gnode : graph->get_ordered_ops())
    {
        if (async_node.find(gnode->get_op_type()) != async_node.end())
        {
            (*gnode)["is_async_node"] = true;
            num_async_node += 1;
        }
    }

    if (num_async_node == 0 && num_gpu == 1)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            if (!async_info.execution_thread)
                async_info.execution_thread = host_async_manager->set_stream(0, "default");
        }
    }
    else
    {
        //const
        set<string> constant_vals;

        int count = 0;
        int max_num_async_thread = 1000; // could be modified.

        for (auto gnode : graph->get_ordered_ops())
        {
            auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            if (async_info.execution_thread)
                continue;
            if ((*gnode)["is_async_node"].is_valid_as<bool>())
            {
                async_info.execution_thread =
                    host_async_manager->set_stream(0, "async_" + to_string(count));
                if (count < max_num_async_thread)
                    count += 1;

                for (auto& out_edge : gnode->get_out_edges())
                {
                    auto out_gnode = out_edge->get_dst();
                    auto& out_gnode_async_info =
                        (*out_gnode)["Async_info"].as<AsyncExecutionInfo>();
                    if (!out_gnode_async_info.execution_thread &&
                        out_gnode->get_op_ptr()->is_output())
                        out_gnode_async_info.execution_thread = async_info.execution_thread;
                }
                // std::vector<std::shared_ptr<nnfusion::graph::GNode>> stack;
                // std::unordered_set<std::shared_ptr<nnfusion::graph::GNode>> same_thread{gnode};

                // auto add_gnode = [&stack, &same_thread](std::shared_ptr<GNode> out_gnode) {
                //     if (!(*out_gnode)["Async_info"].is_valid())
                //         (*out_gnode)["Async_info"] = AsyncExecutionInfo();
                //     auto& out_async_info = (*out_gnode)["Async_info"].as<AsyncExecutionInfo>();
                //     bool same = true;
                //     for (auto& in_edge : out_gnode->get_in_edges())
                //     {
                //         auto in_node = in_edge->get_src();
                //         if (same_thread.find(in_node) == same_thread.end())
                //         {
                //             same = false;
                //             break;
                //         }
                //     }
                //     if (same && out_async_info.execution_thread == nullptr &&
                //         !(*out_gnode)["is_async_node"].is_valid())
                //     {
                //         stack.push_back(out_gnode);
                //         same_thread.insert(out_gnode);
                //     }
                // };

                // for (auto& out_edge : gnode->get_out_edges())
                // {
                //     auto out_gnode = out_edge->get_dst();
                //     int out_device_id = (*out_gnode)["DeviceID"].as<int>();
                //     auto out_device_type = (*out_gnode)["DeviceType"].as<NNFusion_DeviceType>();
                //     if (out_device_id == device_id && out_device_type == device_type)
                //         add_gnode(out_edge->get_dst());
                // }

                // while (!stack.empty())
                // {
                //     std::shared_ptr<GNode> cur_gnode = stack.back();
                //     stack.pop_back();
                //     auto& cur_async_info = (*cur_gnode)["Async_info"].as<AsyncExecutionInfo>();
                //     cur_async_info.execution_thread = async_info.execution_thread;
                //     for (auto& out_edge : cur_gnode->get_out_edges())
                //     {
                //         auto out_gnode = out_edge->get_dst();
                //         int out_device_id = (*out_gnode)["DeviceID"].as<int>();
                //         if (out_device_id == device_id)
                //             add_gnode(out_edge->get_dst());
                //     }
                // }
            }
            else
            {
                // for (auto& in_edge : gnode->get_in_edges())
                // {
                //     auto in_gnode = in_edge->get_src();
                //     int in_device_id = (*in_gnode)["DeviceID"].as<int>();
                //     if (in_device_id == device_id && !in_gnode->get_op_ptr()->is_tensor_op())
                //     {
                //         auto& in_async_info = (*in_gnode)["Async_info"].as<AsyncExecutionInfo>();
                //         auto in_thread = in_async_info.execution_thread;
                //         if (in_thread != nullptr)
                //         {
                //             async_info.execution_thread = in_thread;
                //             break;
                //         }
                //     }
                // }

                if (async_info.execution_thread == nullptr)
                {
                    async_info.execution_thread =
                        host_async_manager->set_stream(0, "dev" + to_string(device_id));
                }
            }
        }
    }

    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

// assign thread of cpu
void AssignAsyncInfoPass::naive_assign_thread_info(shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 0)
        n_stream = 1;
    if (n_stream == 1)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            if (!async_info.execution_thread)
                async_info.execution_thread = async_manager->set_stream(0, "default");
        }
    }
    else
    {
        //const
        set<string> constant_vals;

        int count = 1;
        const GNodeVector& start = graph->get_outputs();
        // Stack of work to do.
        std::vector<std::shared_ptr<GNode>> stack(start.size());

        for (int i = 0; i < start.size(); ++i)
        {
            stack[i] = start[i];
        }

        std::vector<bool> visited(graph->get_max_node_id(), false);
        while (!stack.empty())
        {
            std::shared_ptr<GNode> gnode = stack.back();
            stack.pop_back();
            if (visited[gnode->get_id()])
            {
                continue;
            }
            visited[gnode->get_id()] = true;
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();

            if (async_info.execution_thread == nullptr)
            {
                async_info.execution_thread =
                    async_manager->set_stream(0, "_" + std::to_string(count));
            }

            auto add_gnode = [&visited, &stack](std::shared_ptr<GNode> in_node) {
                if (!visited[in_node->get_id()])
                {
                    // Note; we must not mark as visited until we actually process it.
                    stack.push_back(in_node);
                }
            };

            size_t pre = stack.size();

            for (auto in_edge : gnode->get_in_edges())
            {
                add_gnode(in_edge->get_src());
            }

            if (stack.size() == pre)
            {
                if (n_stream == 0 || count < n_stream)
                    count += 1;
                else
                    count = 1;
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

// assign stream of gpu or thread of cpu
void AssignAsyncInfoPass::naive_assign_stream_info(shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_device_stream_async_manager(graph, CUDA_GPU);
    bool allreduce_enable = FLAGS_fadd_allreduce;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 1)
        n_stream = 1;

    if (n_stream == 1 && FLAGS_fuse_default_stream)
    {
        for (auto gnode : graph->get_ordered_ops())
        {
            NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
            if (async_info.execution_stream || (device_type != CUDA_GPU && device_type != ROCM_GPU))
                continue;
            auto thread = async_info.execution_thread;
            NNFUSION_CHECK(thread != nullptr);
            if (thread->is_default_stream())
            {
                async_info.execution_stream = async_manager->set_stream(device_id, "default");
            }
            else
            {
                std::string thread_name = thread->get_name();
                async_info.execution_stream = async_manager->set_stream(device_id, thread_name);
            }
        }
    }
    else
    {
        //const
        set<string> constant_vals;

        int count = 1;
        const GNodeVector& start = graph->get_outputs();
        // Stack of work to do.
        std::vector<std::shared_ptr<GNode>> stack(start.size());

        for (int i = 0; i < start.size(); ++i)
        {
            stack[i] = start[i];
        }

        std::vector<bool> visited(graph->get_max_node_id(), false);
        while (!stack.empty())
        {
            std::shared_ptr<GNode> gnode = stack.back();
            stack.pop_back();
            if (visited[gnode->get_id()])
            {
                continue;
            }
            visited[gnode->get_id()] = true;
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
            if (!async_info.execution_stream &&
                (device_type == CUDA_GPU || device_type == ROCM_GPU))
            {
                auto thread = async_info.execution_thread;
                NNFUSION_CHECK(thread != nullptr);
                std::string thread_name = thread->get_name();
                async_info.execution_stream =
                    async_manager->set_stream(device_id, thread_name + "_" + std::to_string(count));
            }
            auto add_gnode = [&visited, &stack](std::shared_ptr<GNode> in_node) {
                if (!visited[in_node->get_id()])
                {
                    // Note; we must not mark as visited until we actually process it.
                    stack.push_back(in_node);
                }
            };

            size_t pre = stack.size();

            for (auto in_edge : gnode->get_in_edges())
            {
                add_gnode(in_edge->get_src());
            }

            if (stack.size() == pre)
            {
                if (n_stream == 0 || count < n_stream)
                    count += 1;
                else
                    count = 1;
            }
        }
    }
    // add binding info
    for (auto gnode : graph->get_ordered_ops())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto kernel = get_kernel(gnode);
        if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(get_kernel(gnode)))
        {
            auto stream = async_info.execution_stream;
            if (kernel->require_cudnn_handle())
                stream->add_binding_symbol("cudnn_handle");
            if (kernel->require_cublas_handle())
                stream->add_binding_symbol("cublas_handle");
        }
    }
    NNFUSION_LOG(INFO) << "assign stream info-------------------------------";
}

// assign event of gpu or barrier of cpu
void AssignAsyncInfoPass::assign_event_info(std::shared_ptr<Graph>& graph)
{
    auto host_async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    auto CUDA_async_manager = AsyncManagerFactory::get_device_stream_async_manager(graph, CUDA_GPU);
    for (auto gnode : graph->get_ordered_ops())
    {
        NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto stream = async_info.execution_stream;
        auto thread = async_info.execution_thread;

        for (auto& edge : gnode->get_in_edges())
        {
            auto input_gnode = edge->get_src();
            // constant and rt_const_folding ops are in xxx_init() of generated code,
            // so there is no need to add event.
            if (input_gnode->get_op_ptr()->is_tensor_op() ||
                (*input_gnode)["rt_const_folding"].is_valid_as<bool>())
            {
                continue;
            }
            auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
            auto input_stream = input_async_info.execution_stream;
            auto input_thread = input_async_info.execution_thread;

            if (input_thread->get_stream_id() != thread->get_stream_id())
            {
                if (input_async_info.notify_barrier == nullptr)
                {
                    input_async_info.notify_barrier = host_async_manager->set_event(
                        input_thread, input_gnode->get_op_ptr()->get_unique_name());
                }
                async_info.wait_barriers.push_back(input_async_info.notify_barrier);
            }

            if (stream != nullptr && input_stream != nullptr)
            {
                if (input_stream->get_stream_id() != stream->get_stream_id())
                {
                    // Cuda streams perform implicite sychronization with default(0) stream,
                    // so there is no need to add event emplicitely.
                    if (stream->is_default_stream() || input_stream->is_default_stream())
                    {
                        continue;
                    }
                    if (input_async_info.record_event == nullptr)
                    {
                        input_async_info.record_event = CUDA_async_manager->set_event(
                            input_stream, input_gnode->get_op_ptr()->get_unique_name());
                    }
                    async_info.wait_events.push_back(input_async_info.record_event);
                }
            }
        }
    }
    NNFUSION_LOG(INFO) << "assign event info-------------------------------";
}

void AssignAsyncInfoPass::init_assign_async_info(std::shared_ptr<Graph>& graph)
{
    auto host_async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    auto CUDA_async_manager = AsyncManagerFactory::get_device_stream_async_manager(graph, CUDA_GPU);
    set<string> constant_vals;
    bool use_bfs = false;
    nnfusion::graph::GNodeVector node_vec;
    if (FLAGS_fstream_assign_policy == "kernel_prof_based")
        node_vec = graph->get_bfs_ordered_ops();
    else
        node_vec = graph->get_ordered_ops();
    for (auto gnode : node_vec)
    {
        (*gnode)["Async_info"] = AsyncExecutionInfo();
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        NNFUSION_CHECK((*gnode)["DeviceID"].is_valid());
        NNFUSION_CHECK((*gnode)["DeviceType"].is_valid());
        int device_id = (*gnode)["DeviceID"].as<int>();
        auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
        if (gnode->get_op_ptr()->is_tensor_op())
        {
            async_info.execution_thread = host_async_manager->set_stream(0, "default");
            if (device_type == CUDA_GPU || device_type == ROCM_GPU)
            {
                async_info.execution_stream =
                    CUDA_async_manager->set_stream(device_id, FLAGS_fcuda_init_stream);
            }
            if (FLAGS_frt_const_folding && gnode->is_constant())
            {
                auto kernel = get_kernel(gnode);
                NNFUSION_CHECK_NOT_NULLPTR(kernel);
                (*gnode)["rt_const_folding"] = true;
                for (auto& out : kernel->m_context->output_names)
                    constant_vals.insert(out);
            }
        }
        else
        {
            if (FLAGS_frt_const_folding)
            {
                bool const_inputs = true;
                if (auto kernel = get_kernel(gnode))
                {
                    for (auto& in : kernel->m_context->input_names)
                    {
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }

                    if (const_inputs && !gnode->get_op_ptr()->is_output())
                    {
                        (*gnode)["rt_const_folding"] = true;
                        for (auto& out : kernel->m_context->output_names)
                        {
                            constant_vals.insert(out);
                        }
                        async_info.execution_thread = host_async_manager->set_stream(0, "default");
                        if (device_type == CUDA_GPU || device_type == ROCM_GPU)
                        {
                            async_info.execution_stream =
                                CUDA_async_manager->set_stream(device_id, FLAGS_fcuda_init_stream);
                        }
                    }
                }
            }
        }
    }
}

// for single thread and single device only
void AssignAsyncInfoPass::kernel_prof_based_assign_stream_info(std::shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_device_stream_async_manager(graph, CUDA_GPU);
    bool allreduce_enable = FLAGS_fadd_allreduce;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 1)
        n_stream = 1;

    if (n_stream == 1 && FLAGS_fuse_default_stream)
    {
        for (auto gnode : graph->get_bfs_ordered_ops())
        {
            NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
            if (async_info.execution_stream || (device_type != CUDA_GPU && device_type != ROCM_GPU))
                continue;
            auto thread = async_info.execution_thread;
            NNFUSION_CHECK(thread != nullptr);
            if (thread->is_default_stream())
            {
                async_info.execution_stream = async_manager->set_stream(device_id, "default");
            }
            else
            {
                std::string thread_name = thread->get_name();
                async_info.execution_stream = async_manager->set_stream(device_id, thread_name);
            }
        }
    }
    else
    {
        //assign stream and event
        std::unordered_map<std::shared_ptr<Stream>, uint64_t> stream_time;
        std::unordered_map<std::shared_ptr<Stream>, shared_ptr<GNode>> stream_tail_nodes;
        // relative time point after exec the kernel of gnode
        std::unordered_map<std::shared_ptr<GNode>, uint64_t> gnode_time;

        for (auto gnode : graph->get_bfs_ordered_ops())
        {
            NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();

            if (async_info.execution_stream)
                continue;
            // if the tail node of stream S is the direct dependent node of gnode N,
            // assign stream S to gnode N.
            for (auto& edge : gnode->get_in_edges())
            {
                auto input_gnode = edge->get_src();
                auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                NNFUSION_CHECK_NOT_NULLPTR(input_async_info.execution_stream);

                bool input_is_tail_node = false;
                for (auto s_tn : stream_tail_nodes)
                {
                    if (s_tn.second->get_instance_id() == input_gnode->get_instance_id())
                    {
                        input_is_tail_node = true;
                        break;
                    }
                }
                if (input_is_tail_node)
                {
                    async_info.execution_stream = input_async_info.execution_stream;
                    stream_tail_nodes[async_info.execution_stream] = gnode;
                    break;
                }
            }
            if (!async_info.execution_stream)
            {
                uint64_t p_max_time = 0;
                uint64_t p_min_time = numeric_limits<uint64_t>::max();
                std::shared_ptr<Stream> cand_thread;
                for (auto& edge : gnode->get_in_edges())
                {
                    auto input_gnode = edge->get_src();
                    auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                    NNFUSION_CHECK_NOT_NULLPTR(input_async_info.execution_stream);
                    if (gnode_time[input_gnode] > p_max_time &&
                        stream_time.find(input_async_info.execution_stream) != stream_time.end())
                    {
                        p_max_time = gnode_time[input_gnode];
                        cand_thread = input_async_info.execution_stream;
                        p_min_time = stream_time[input_async_info.execution_thread];
                    }
                }
                if (cand_thread)
                {
                    for (auto s_t : stream_time)
                    {
                        if (s_t.second > p_max_time + 10 && s_t.second < p_min_time)
                        {
                            p_min_time = s_t.second;
                            cand_thread = s_t.first;
                        }
                    }
                    async_info.execution_stream = cand_thread;
                    stream_tail_nodes[cand_thread] = gnode;
                }
                // else assign the shortest stream to gnode N
                else
                {
                    if (stream_time.size() < n_stream)
                    {
                        auto thread = async_info.execution_thread;
                        NNFUSION_CHECK(thread != nullptr);
                        std::string thread_name = thread->get_name();
                        async_info.execution_stream = async_manager->set_stream(
                            device_id, thread_name + "_" + to_string(stream_time.size()));
                        stream_tail_nodes[async_info.execution_stream] = gnode;
                    }
                    else
                    {
                        uint64_t min_time = numeric_limits<uint64_t>::max();
                        for (auto s_t : stream_time)
                        {
                            if (s_t.second < min_time)
                            {
                                min_time = s_t.second;
                                async_info.execution_stream = s_t.first;
                            }
                        }
                        stream_tail_nodes[async_info.execution_stream] = gnode;
                    }
                }
            }
            // add event
            auto cur_stream = async_info.execution_stream;
            NNFUSION_CHECK_NOT_NULLPTR(cur_stream);

            uint64_t max_input_time = 0;

            for (auto& edge : gnode->get_in_edges())
            {
                auto input_gnode = edge->get_src();
                // constant and rt_const_folding ops are in xxx_init() of generated code,
                // so there is no need to add event.
                if (input_gnode->get_op_ptr()->is_tensor_op() ||
                    (*input_gnode)["rt_const_folding"].is_valid_as<bool>())
                {
                    continue;
                }
                auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                auto input_stream = input_async_info.execution_stream;
                NNFUSION_CHECK_NOT_NULLPTR(input_stream);
                if (input_stream->get_stream_id() != cur_stream->get_stream_id())
                {
                    // Cuda streams perform implicite sychronization with default(0) stream,
                    // so there is no need to add event emplicitely.
                    if (cur_stream->is_default_stream() || input_stream->is_default_stream())
                    {
                        continue;
                    }
                    if (input_async_info.record_event == nullptr)
                    {
                        input_async_info.record_event = async_manager->set_event(
                            input_stream, input_gnode->get_op_ptr()->get_unique_name());
                    }
                    async_info.wait_events.push_back(input_async_info.record_event);
                    if (gnode_time.find(input_gnode) != gnode_time.end() &&
                        gnode_time[input_gnode] > max_input_time)
                        max_input_time = gnode_time[input_gnode];
                }
            }
            // compute time cost
            if (max_input_time > stream_time[cur_stream])
                gnode_time[gnode] = get_time_cost(gnode) + max_input_time;
            else
                gnode_time[gnode] = get_time_cost(gnode) + stream_time[cur_stream];

            stream_time[cur_stream] = gnode_time[gnode];
        }
        for (auto info : stream_time)
        {
            NNFUSION_LOG(INFO) << info.first->get_name() << ":" << to_string(info.second);
        }
    }
    // add binding info
    for (auto gnode : graph->get_bfs_ordered_ops())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto kernel = get_kernel(gnode);
        if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(get_kernel(gnode)))
        {
            auto stream = async_info.execution_stream;
            if (kernel->require_cudnn_handle())
                stream->add_binding_symbol("cudnn_handle");
            if (kernel->require_cublas_handle())
                stream->add_binding_symbol("cublas_handle");
        }
    }
    NNFUSION_LOG(INFO) << "assign stream info-------------------------------";
}

void AssignAsyncInfoPass::kernel_prof_based_assign_thread_info(std::shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    bool allreduce_enable = FLAGS_fadd_allreduce;
    int n_stream = FLAGS_fnum_stream;
    if (n_stream < 1)
        n_stream = 1;

    if (n_stream == 1 && FLAGS_fuse_default_stream)
    {
        for (auto gnode : graph->get_bfs_ordered_ops())
        {
            if (!(*gnode)["Async_info"].is_valid())
                (*gnode)["Async_info"] = AsyncExecutionInfo();
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*gnode)["DeviceID"].as<int>();
            if (!async_info.execution_thread)
                async_info.execution_thread = async_manager->set_stream(0, "default");
        }
    }
    else
    {
        //assign stream and event
        std::unordered_map<std::shared_ptr<Stream>, uint64_t> stream_time;
        std::unordered_map<std::shared_ptr<Stream>, shared_ptr<GNode>> stream_tail_nodes;
        // relative time point after exec the kernel of gnode
        std::unordered_map<std::shared_ptr<GNode>, uint64_t> gnode_time;

        for (auto gnode : graph->get_bfs_ordered_ops())
        {
            NNFUSION_CHECK((*gnode)["Async_info"].is_valid());
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            if (async_info.execution_thread)
                continue;
            int device_id = (*gnode)["DeviceID"].as<int>();
            // if the tail node of stream S is the direct dependent node of gnode N,
            // assign stream S to gnode N.

            for (auto& edge : gnode->get_in_edges())
            {
                auto input_gnode = edge->get_src();
                auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                NNFUSION_CHECK_NOT_NULLPTR(input_async_info.execution_thread);

                bool input_is_tail_node = false;
                for (auto s_tn : stream_tail_nodes)
                {
                    if (s_tn.second->get_instance_id() == input_gnode->get_instance_id())
                    {
                        input_is_tail_node = true;
                        break;
                    }
                }
                if (input_is_tail_node)
                {
                    async_info.execution_thread = input_async_info.execution_thread;
                    stream_tail_nodes[async_info.execution_thread] = gnode;
                    break;
                }
            }

            if (!async_info.execution_thread)
            {
                uint64_t p_max_time = 0.0;
                uint64_t p_min_time = numeric_limits<uint64_t>::max();
                std::shared_ptr<Stream> cand_thread;
                for (auto& edge : gnode->get_in_edges())
                {
                    auto input_gnode = edge->get_src();
                    auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                    NNFUSION_CHECK_NOT_NULLPTR(input_async_info.execution_thread);
                    if (gnode_time[input_gnode] > p_max_time &&
                        stream_time.find(input_async_info.execution_thread) != stream_time.end())
                    {
                        p_max_time = gnode_time[input_gnode];
                        cand_thread = input_async_info.execution_thread;
                        p_min_time = stream_time[input_async_info.execution_thread];
                    }
                }

                if (cand_thread)
                {
                    for (auto s_t : stream_time)
                    {
                        if (s_t.second > p_max_time + 10 && s_t.second < p_min_time)
                        {
                            p_min_time = s_t.second;
                            cand_thread = s_t.first;
                        }
                    }
                    async_info.execution_thread = cand_thread;
                    stream_tail_nodes[cand_thread] = gnode;
                }
                // else assign the shortest stream to gnode N
                else
                {
                    if (stream_time.size() < n_stream)
                    {
                        async_info.execution_thread = async_manager->set_stream(
                            device_id, "base" + to_string(stream_time.size()));
                        stream_tail_nodes[async_info.execution_thread] = gnode;
                    }
                    else
                    {
                        uint64_t min_time = numeric_limits<uint64_t>::max();
                        for (auto s_t : stream_time)
                        {
                            if (s_t.second < min_time)
                            {
                                min_time = s_t.second;
                                async_info.execution_thread = s_t.first;
                            }
                        }
                        stream_tail_nodes[async_info.execution_thread] = gnode;
                    }
                }
            }
            // add event
            auto cur_stream = async_info.execution_thread;
            NNFUSION_CHECK_NOT_NULLPTR(cur_stream);

            uint64_t max_input_time = 0;
            for (auto& edge : gnode->get_in_edges())
            {
                auto input_gnode = edge->get_src();
                // constant and rt_const_folding ops are in xxx_init() of generated code,
                // so there is no need to add event.
                if (input_gnode->get_op_ptr()->is_tensor_op() ||
                    (*input_gnode)["rt_const_folding"].is_valid_as<bool>())
                {
                    continue;
                }
                auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
                auto input_stream = input_async_info.execution_thread;
                NNFUSION_CHECK_NOT_NULLPTR(input_stream);
                if (input_stream->get_stream_id() != cur_stream->get_stream_id())
                {
                    if (input_async_info.notify_barrier == nullptr)
                    {
                        input_async_info.notify_barrier = async_manager->set_event(
                            input_stream, input_gnode->get_op_ptr()->get_unique_name());
                    }
                    async_info.wait_barriers.push_back(input_async_info.notify_barrier);
                    if (gnode_time.find(input_gnode) != gnode_time.end() &&
                        gnode_time[input_gnode] >= max_input_time)
                        max_input_time = gnode_time[input_gnode];
                }
            }
            // compute time cost
            if (max_input_time > stream_time[cur_stream])
                gnode_time[gnode] = get_time_cost(gnode) + max_input_time;
            else
                gnode_time[gnode] = get_time_cost(gnode) + stream_time[cur_stream];

            stream_time[cur_stream] = gnode_time[gnode];
        }
        for (auto info : stream_time)
        {
            NNFUSION_LOG(INFO) << info.first->get_name() << ": " << to_string(info.second) << " us";
        }
    }

    NNFUSION_LOG(INFO) << "assign thread info-------------------------------";
}

void AssignAsyncInfoPass::assign_default_info(std::shared_ptr<Graph>& graph)
{
    auto host_async_manager = AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    auto device_async_manager = AsyncManagerFactory::get_device_stream_async_manager(
        graph, nnfusion::get_device_type(FLAGS_fdefault_device));
    for (auto gnode : graph->get_nodes())
    {
        (*gnode)["Async_info"] = AsyncExecutionInfo();
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        NNFUSION_CHECK((*gnode)["DeviceID"].is_valid());
        NNFUSION_CHECK((*gnode)["DeviceType"].is_valid());
        int device_id = (*gnode)["DeviceID"].as<int>();
        auto device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
        async_info.execution_thread = host_async_manager->set_stream(device_id, "default");
        async_info.execution_stream = device_async_manager->set_stream(0, "default");
    }
}

KernelEmitter::Pointer
    AssignAsyncInfoPass::get_kernel(std::shared_ptr<nnfusion::graph::GNode> gnode)
{
    KernelEmitter::Pointer kernel = nullptr;
    if (!gnode->is_parameter())
        NNFUSION_CHECK((*gnode)["Kernel_Selection_Result"].is_valid())
            << "Kernel should be selected before this pass:" << gnode->get_op_type();
    if ((*gnode)["Kernel_Selection_Result"].is_valid())
    {
        auto emitted_kernel = (*gnode)["Kernel_Selection_Result"]
                                  .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();

        if (!gnode->is_constant() && !emitted_kernel.second->is_emitted())
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                           << gnode->get_op_type();
        kernel = emitted_kernel.second;
    }

    return kernel;
}

uint64_t AssignAsyncInfoPass::get_time_cost(std::shared_ptr<nnfusion::graph::GNode> gnode)
{
    if ((*gnode)["Kernel_Selection_Result"].is_valid() &&
        (*gnode)["Kernel_Selection_Result"]
            .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>()
            .second->is_eliminative())

    {
        return 0;
    }
    nnfusion::pass::graph::KernelProfilingRecord::Pointer result;
    if ((*gnode)["Kernel_Profiling_Result"].is_valid())
    {
        result = (*gnode)["Kernel_Profiling_Result"]
                     .as<nnfusion::pass::graph::KernelProfilingRecord::Pointer>();
    }
    if (result && result->valid)
    {
        return result->kernel_time_in_us;
    }
    else
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be profiled before this pass"
                                       << gnode->get_name() << "\t" << gnode->get_op_type();
        return 0;
    }
}
