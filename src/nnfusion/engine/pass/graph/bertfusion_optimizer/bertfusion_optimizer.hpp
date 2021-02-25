// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <queue>
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/util/util.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            struct BertFusionGroup
            {
                std::unordered_map<std::string,
                                   std::vector<std::shared_ptr<nnfusion::graph::GNode>>>
                    fuse_group;
                std::unordered_set<std::shared_ptr<nnfusion::graph::GNode>> nodes_to_remove;
                std::vector<std::shared_ptr<nnfusion::graph::GNode>> edge_nodes;
                std::vector<std::shared_ptr<nnfusion::graph::GNode>> helper_nodes;
            };

            class BertFusionOptimizer
            {
            public:
                BertFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> g)
                    : m_graph(g)
                {
                }

                virtual bool Optimize();

            protected:
                virtual bool CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node) = 0;
                virtual bool FindSubGraph(std::shared_ptr<nnfusion::graph::GNode> starting_node,
                                          std::shared_ptr<BertFusionGroup> bertfusion_group) = 0;
                virtual bool FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group) = 0;
                bool RemoveNodes(std::unordered_set<std::shared_ptr<nnfusion::graph::GNode>> nodes,
                                 std::shared_ptr<nnfusion::graph::GNode> new_node);
                virtual bool FindPath(
                    std::shared_ptr<nnfusion::graph::GNode> node,
                    std::vector<std::string>& pattern,
                    std::vector<std::vector<std::shared_ptr<nnfusion::graph::GNode>>>& all_paths,
                    bool reverse);
                virtual void Search(
                    std::shared_ptr<nnfusion::graph::GNode> node,
                    std::vector<std::string>& pattern,
                    size_t idx,
                    std::vector<std::vector<std::shared_ptr<nnfusion::graph::GNode>>>& all_paths,
                    std::vector<std::shared_ptr<nnfusion::graph::GNode>>& path,
                    bool reverse);
                bool update_graph_outputs(
                    std::unordered_set<std::shared_ptr<nnfusion::graph::GNode>>& nodes_to_remove,
                    std::shared_ptr<nnfusion::graph::GNode> new_node);
                std::shared_ptr<nnfusion::graph::Graph> m_graph;
            };
        }
    }
}