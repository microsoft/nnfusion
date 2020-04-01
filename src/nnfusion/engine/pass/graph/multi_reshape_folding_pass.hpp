// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph;

DEFINE_bool(ffold_reshape_op, true, "Folding Reshape operators.");

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class MultiReshapeFoldingPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    bool using_pass = FLAGS_ffold_reshape_op;
                    if (!using_pass)
                        return true;

                    std::unordered_set<std::shared_ptr<GNode>> tail_reshape_nodes;

                    auto is_transpose = [](const std::shared_ptr<GNode>& gnode) -> bool {
                        return std::dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr())
                            ->get_is_transpose();
                    };

                    // Find tail nodes exactly after reshape node
                    for (auto& it : graph->get_nodes())
                    {
                        if (it->get_op_type() != "Reshape")
                        {
                            for (auto edge : it->get_in_edges())
                            {
                                if (edge->is_control_edge())
                                    continue;
                                if (edge->get_src()->get_op_type() == "Reshape")
                                {
                                    if (!is_transpose(edge->get_src()))
                                        continue;
                                    tail_reshape_nodes.insert(edge->get_src());
                                }
                            }
                        }
                    }

                    for (auto tail_node : tail_reshape_nodes)
                    {
                        std::vector<std::shared_ptr<GNode>> chain;
                        auto node = tail_node;
                        CHECK_NOT_NULLPTR(node);
                        CHECK(node->get_op_type() == "Reshape");
                        chain.push_back(node);
                        while (true)
                        {
                            if (node->get_in_edges().size() != 1)
                            {
                                break;
                            }
                            auto src = node->get_in_edge(0)->get_src();
                            if (src->get_op_type() != "Reshape" || !is_transpose(src) ||
                                src->get_out_edges().size() != 1)
                            {
                                break;
                            }
                            node = src;
                            chain.push_back(node);
                        }

                        if (chain.size() <= 1)
                            continue;

                        AxisVector order, mirror;
                        auto rs_inedge = node->get_in_edge(0);
                        auto rs_input = rs_inedge->get_src();
                        for (int i = 0;
                             i < rs_input->get_output_shape(rs_inedge->get_src_output()).size();
                             ++i)
                        {
                            order.push_back(i);
                        }

                        for (int i = chain.size() - 1; i >= 0; --i)
                        {
                            auto chord =
                                std::dynamic_pointer_cast<op::Reshape>(chain[i]->get_op_ptr())
                                    ->get_input_order();
                            CHECK(order.size() == chord.size());
                            mirror.resize(order.size());
                            for (int i = 0; i < chord.size(); ++i)
                                mirror[i] = order[chord[i]];
                            order = std::move(mirror);
                        }
                        auto top_shape = rs_input->get_output_shape(rs_inedge->get_src_output());
                        auto out_shape = top_shape;
                        CHECK(top_shape.size() == order.size());
                        for (int i = 0; i < top_shape.size(); ++i)
                        {
                            out_shape[i] = top_shape[order[i]];
                        }

                        auto reshape_op = std::make_shared<op::Reshape>(order, out_shape);
                        auto reshape_gnode = graph->add_node_and_edge(
                            reshape_op,
                            GNodeIndexVector({GNodeIndex(rs_input, rs_inedge->get_src_output())}));

                        auto tail_reshape = chain[0];
                        for (auto edge : tail_reshape->get_out_edges())
                        {
                            graph->add_edge(reshape_gnode,
                                            edge->get_src_output(),
                                            edge->get_dst(),
                                            edge->get_dst_input());
                        }

                        for (auto node : chain)
                        {
                            graph->remove_node(node);
                        }
                    }

                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
