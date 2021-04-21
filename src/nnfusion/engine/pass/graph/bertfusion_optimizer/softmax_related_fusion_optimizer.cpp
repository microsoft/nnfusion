// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax_related_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool SoftmaxRelatedFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_op_type() == "Softmax")
        return true;
    return false;
}


bool SoftmaxRelatedFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                       std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    //reshape, broadcast, divide, add, softmax
    auto softmax = starting_node;
    auto add = softmax->get_in_edge(0)->get_src();
    auto div = add->get_in_edge(0)->get_src();
    auto reshape = div->get_in_edge(0)->get_src();
    auto broadcast = div->get_in_edge(1)->get_src();

    bertfusion_group->fuse_group["inputs"] = {
        broadcast, reshape, div, add, softmax
    };
    bertfusion_group->nodes_to_remove.insert({
        broadcast, reshape, div, add, softmax
    });
    
    bertfusion_group->helper_nodes.push_back(starting_node);
    bertfusion_group->edge_nodes.push_back(softmax);
    return true;
}

bool SoftmaxRelatedFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 5);

    // fused = softmax(reshape(input0)/input1 + broadcast(input2));
    auto broadcast = inputs[0];
    auto reshape = inputs[1];
    auto div = inputs[2];
    auto add = inputs[3];
    auto softmax = inputs[4];

    auto input = reshape->get_in_edge(0)->get_src();
    auto input_scale = broadcast ->get_in_edge(0)->get_src();
    auto bias = add->get_in_edge(1)->get_src();

    nnfusion::op::OpConfig::any myConfig;
    auto fused_softmax = std::make_shared<nnfusion::op::GenericOp>(
        "Fused_Div_Add_Softmax_" + input->get_name() + "_ " + input_scale->get_name(),
            "FusedDivAddSoftmax", myConfig);
    
    auto fused_softmax_node = m_graph->add_node_and_edge(fused_softmax,
                                                           {
                                                               GNodeIndex{input, 0},
                                                               GNodeIndex{input_scale, 0},
                                                               GNodeIndex{bias, 0},
                                                           }
                                                            );
    fused_softmax_node->set_output_type_and_shape(
        0, fused_softmax_node->get_input_element_type(0), softmax->get_input_shape(0));

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(fused_softmax_node, 0, dst, y);
        }
    }
    return RemoveNodes(bertfusion_group->nodes_to_remove, fused_softmax_node);
}
