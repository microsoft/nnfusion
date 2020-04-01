// Microsoft (c) 2019, NNFusion Team
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
            class DefaultDeviceDispatcher : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };

        } // namespace graph
    }     // namespace pass
} // namespace nnfusion