// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "subgraph_fusion_optimizer.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class MatMulAddFusionOptimizer : public SubGraphFusionOptimizer
            {
            public:
                MatMulAddFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> graph)
                    : SubGraphFusionOptimizer(graph)

                {
                }
                virtual bool create_subgraphs() override;
                virtual bool fuse_subgraph(SubGraphRecord::Pointer subgraph_record) override;
            };
        } // namespace graph
    }     // namespace pass
} // namespace nnfusion