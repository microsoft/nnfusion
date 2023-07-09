// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            struct KernelProfilingRecord
            {
                uint64_t kernel_time_in_us;
                bool valid = false;
                // more profiling info can be added here
                using Pointer = shared_ptr<KernelProfilingRecord>;
            };

            class KernelProfilingPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                bool default_profiling_pass(std::shared_ptr<nnfusion::graph::Graph>& graph);
                bool merged_profiling_pass(std::shared_ptr<nnfusion::graph::Graph>& graph);
            };
        }
    }

} // namespace nnfusion