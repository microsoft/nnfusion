// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphPassBase
            {
            public:
                virtual bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) = 0;
            };
        }
    }
}
