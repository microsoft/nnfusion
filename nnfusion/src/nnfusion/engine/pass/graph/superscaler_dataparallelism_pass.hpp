// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include "graph_pass_base.hpp"
using namespace nnfusion::graph;

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class SuperScalerDataParallelismPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                std::shared_ptr<GNode> concat_into_one(
                    std::shared_ptr<Graph>& graph,
                    std::vector<int> subgroup,
                    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>>
                        hash_to_gradient_apply);
                std::vector<std::pair<std::shared_ptr<GNode>, int>> split_from_one(
                    std::shared_ptr<Graph>& graph,
                    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>>
                        hash_to_gradient_apply,
                    std::shared_ptr<GNode> allreduce_node,
                    std::vector<int> subgroup);
                bool add_allreduce(
                    std::shared_ptr<Graph>& graph,
                    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>>
                        hash_to_gradient_apply);
                bool add_fused_allreduce(
                    std::shared_ptr<Graph>& graph,
                    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>>
                        hash_to_gradient_apply);
                std::vector<std::vector<int>> group_gradient_apply(
                    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>>
                        hash_to_gradient_apply);
                int get_gradient_from_apply(std::shared_ptr<GNode> apply_node);
                bool sc_allreduce_enable;
                bool sc_allreduce_fusion_enable;
                template <class T>
                void add_inplace(T op, size_t output, size_t input, bool destructive)
                {
                    auto op_annotations = op->get_op_annotations();
                    if (op_annotations)
                    {
                        // pass-through
                        op_annotations->add_in_place_oi_pair({output, input, destructive});
                    }
                    else
                    {
                        op_annotations = std::make_shared<Annotations>();
                        // pass-through
                        op_annotations->add_in_place_oi_pair({output, input, destructive});
                        op->set_op_annotations(op_annotations);
                    }
                }
            };
        }
    }
}
