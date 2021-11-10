// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <condition_variable>
#include <future>
#include <queue>
#include <thread>

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph;

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class RuntimeConstantFoldingPass : public GraphPassBase
            {
                std::shared_ptr<GNode>
                    runtime_const_folding_node(std::shared_ptr<Graph>& graph,
                                               std::set<std::shared_ptr<GNode>>& blocklist_nodes,
                                               std::shared_ptr<GNode>& node);
                void runtime_const_folding_task(std::shared_ptr<Graph>& graph,
                                                std::set<std::shared_ptr<GNode>>& blocklist_nodes,
                                                std::shared_ptr<GNode> node,
                                                std::map<std::shared_ptr<GNode>, int>& in_degree,
                                                std::mutex& in_degree_lock);

                bool run_on_graph_parallel(std::shared_ptr<Graph>& graph,
                                           std::set<std::shared_ptr<GNode>>& blocklist_nodes);

                class thread_pool
                {
                    using Task = std::function<void()>;
                    std::vector<std::thread> pool;
                    std::queue<Task> tasks;
                    std::mutex m_lock;
                    std::mutex m_lock_done;
                    std::condition_variable cv_task;
                    std::condition_variable cv_task_done;
                    std::atomic<bool> stopped;
                    std::atomic<int> idl_thread_num;
                    int total_thread_num;

                public:
                    thread_pool();
                    ~thread_pool();
                    void commit(Task task);
                    bool is_free();
                    void wait_for_all();
                };

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override;

            private:
                std::string backend;
                bool fast_debug;
                std::shared_ptr<thread_pool> pool_ptr;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
