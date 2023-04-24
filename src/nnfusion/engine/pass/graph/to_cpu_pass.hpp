// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            // Convert the conv operators to CNHW layout for its better performance
            // Insert necessary transpose and modify axis attribute when necessary
            class ToCPUPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
