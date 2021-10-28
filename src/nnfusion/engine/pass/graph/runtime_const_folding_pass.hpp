// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <queue>
#include <condition_variable>
#include <future>

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
                int runtime_const_folding_iterate_once(
                    std::shared_ptr<Graph>& graph,
                    std::set<std::shared_ptr<GNode>>& blocklist_nodes);
                class thread_pool
                {
                    using Task = std::function<void()>;
                    std::vector<std::thread> pool;
                    std::queue<Task> tasks;
                    std::mutex m_lock;
                    std::condition_variable cv_task;
                    std::atomic<bool> stoped;
                    std::atomic<int> idlThrNum;

                public:
                    threadpool(unsigned short size);
                    ~threadpool();
                    template<class F, class... Args>
                    auto commit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;
                    int idlCount();
                }

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override;

            private:
                std::string backend;
                bool fast_debug;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
