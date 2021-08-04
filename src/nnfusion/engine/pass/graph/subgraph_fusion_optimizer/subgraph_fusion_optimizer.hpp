// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <queue>
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/subgraph_match.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class SubGraphFusionOptimizer
            {
            public:
                SubGraphFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> g)
                    : graph(g)
                {
                }

                bool Optimize();

                virtual bool create_subgraphs() = 0;
                bool match_and_fuse_subgraph();
                virtual bool fuse_subgraph(SubGraphRecord::Pointer subgraph_record) = 0;

            protected:
                bool RemoveNodes(std::unordered_set<std::shared_ptr<GNode>>& nodes,
                                 std::shared_ptr<GNode> new_node);
                bool update_graph_outputs(
                    std::unordered_set<std::shared_ptr<GNode>>& nodes_to_remove,
                    std::shared_ptr<GNode> new_node);

                std::vector<SubGraph::Pointer> m_subgraphs;
                std::shared_ptr<SubGraphMatch> subgraph_match;
                std::shared_ptr<nnfusion::graph::Graph> graph;
            };
        }
    }
}