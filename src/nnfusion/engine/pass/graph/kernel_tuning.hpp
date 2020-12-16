// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class KernelTuning : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

                bool register_antares_kernel();

            private:
                std::vector<std::shared_ptr<nnfusion::graph::GNode>>
                    get_tuning_candidates(std::shared_ptr<nnfusion::graph::Graph>& graph);
                bool insert_to_kernel_cache(
                    const std::vector<std::shared_ptr<nnfusion::graph::GNode>>& nodes);

            private:
                static const std::unordered_set<std::string> BlockList;
            };
        }
    }
} // namespace nnfusion