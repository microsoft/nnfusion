// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class PatternSubstitutionPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace graph
    }     // namespace pass
} // namespace nnfusion
