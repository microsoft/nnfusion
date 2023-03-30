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
            class NCHW2NHWCPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                std::shared_ptr<nnfusion::graph::GNode> add_transpose(std::shared_ptr<nnfusion::graph::Graph>& graph,
                    std::shared_ptr<nnfusion::graph::GNode> node, bool to_nhwc);
                void remove_node(std::shared_ptr<nnfusion::graph::Graph>& graph, std::shared_ptr<nnfusion::graph::GNode> node);
            };
        }
    }
}