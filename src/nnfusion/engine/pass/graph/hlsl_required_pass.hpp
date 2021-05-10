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
            // This pass run some necessary change for HLSL codegen, including
            // 1. Convert 64bit integer to 32bit
            class HLSLRequiredPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
