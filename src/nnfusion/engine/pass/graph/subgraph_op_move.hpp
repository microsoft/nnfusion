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
            class SubGraphOpMovePass : public GraphPassBase
            {
            public:
                void move_out(
                    nnfusion::graph::GNodeVector gnodes,
                    std::shared_ptr<nnfusion::graph::GNode> if_node,
                    std::shared_ptr<nnfusion::graph::Graph> full_graph,
                    std::shared_ptr<nnfusion::graph::Graph> graph
                );
                void find_and_move_small_op_out(
                    std::shared_ptr<nnfusion::graph::Graph> graph,
                    std::shared_ptr<nnfusion::graph::Graph> sub_graph,
                    std::shared_ptr<nnfusion::graph::GNode> if_node
                );
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
