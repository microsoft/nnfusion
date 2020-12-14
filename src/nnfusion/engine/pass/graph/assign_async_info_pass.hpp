// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AssignAsyncInfoPass : public GraphPassBase
            {
            public:
                AssignAsyncInfoPass();
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                void gpu_assign_thread_info(std::shared_ptr<Graph>& graph);
                void naive_assign_stream_info(std::shared_ptr<Graph>& graph);
                void naive_assign_thread_info(std::shared_ptr<Graph>& graph);
                void assign_event_info(std::shared_ptr<Graph>& graph);
                void init_assign_async_info(std::shared_ptr<Graph>& graph);
                void kernel_prof_based_assign_stream_info(std::shared_ptr<Graph>& graph);
                void kernel_prof_based_assign_thread_info(std::shared_ptr<Graph>& graph);
                void assign_default_info(std::shared_ptr<Graph>& graph);
                KernelEmitter::Pointer get_kernel(std::shared_ptr<nnfusion::graph::GNode> gnode);
                uint64_t get_time_cost(std::shared_ptr<nnfusion::graph::GNode> gnode);
            };
        }
    }
}