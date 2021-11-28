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
                static void register_single_kernel(const std::string& op_name);
                bool register_antares_kernel();

            private:
                bool parse_block_list();
                bool parse_tuning_list();
                bool insert_to_kernel_cache(
                    const std::vector<std::shared_ptr<nnfusion::graph::GNode>>& nodes);

            private:
                std::unordered_set<std::string> BlockList;
                std::unordered_set<std::string> TuningList;
            };
        }
    }
} // namespace nnfusion