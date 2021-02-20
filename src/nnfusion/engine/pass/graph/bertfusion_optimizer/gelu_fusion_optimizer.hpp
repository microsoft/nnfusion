// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "bertfusion_optimizer.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GeluFusionOptimizer : public BertFusionOptimizer
            {
            public:
                GeluFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> graph)
                    : BertFusionOptimizer(graph)

                {
                }

            private:
                bool CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node) override;
                bool FindSubGraph(std::shared_ptr<nnfusion::graph::GNode> starting_node,
                                  std::shared_ptr<BertFusionGroup> bertfusion_group) override;
                bool FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group) override;
            };

        } // namespace graph
    }     // namespace pass
} // namespace nnfusion
