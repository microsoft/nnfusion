// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "skiplayernorm_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool SkipLayerNormFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_op_type() != "LayerNorm")
        return false;
    if (node->get_in_edge(0)->get_src()->get_op_type() != "Add")
        return false;

    return true;
}

/**
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable

Before fusion:
Format 1:
    [Sub1]  C    [Sub2]
        \  /     /
        Add2    /
           \   /
            Add1
             |
     LayerNormalization

Format 2:
      [Sub1] [Sub2]  C
         \      \   /
          \     Add2
           \    /
            Add1
             |
     LayerNormalization

Format 3:
      [Sub1]   [Sub2]
         \       /
          \     /
           \   /
            Add1
             |
     LayerNormalization

After fusion:
       [Sub1]   [Sub1]
         \      /
          \    /
    SkipLayerNormalization

Note: This fusion doesn't consider the following case:
      [Sub1]   [Sub2]
         \       /
        Add2  Add3
           \   /
            Add1
             |
     LayerNormalization
*/
bool SkipLayerNormFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                                std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);

    // find skip embedding subgraph
    std::vector<std::string> pattern = {starting_node->get_op_type(), "Add", "Add"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths;
    if (!FindPath(starting_node, pattern, all_paths, true) || all_paths.size() != 1)
    {
        auto add1 = starting_node->get_in_edge(0)->get_src();
        auto input = add1->get_in_edge(0)->get_src();
        auto skip = add1->get_in_edge(1)->get_src();
        bertfusion_group->fuse_group["inputs"] = {input, skip};
        bertfusion_group->nodes_to_remove.insert({starting_node, add1});
    }
    else
    {
        auto add1 = all_paths[0][1];
        auto add2 = all_paths[0][2];
        std::shared_ptr<GNode> input, skip;
        if (add1->get_in_edge(0)->get_src() == add2) // format 1
        {
            input = add2->get_in_edge(0)->get_src();
            skip = add1->get_in_edge(1)->get_src();
        }
        else // format 2
        {
            input = add1->get_in_edge(0)->get_src();
            skip = add2->get_in_edge(0)->get_src();
        }

        auto broadcast = add2->get_in_edge(1)->get_src();
        if (broadcast->get_op_type() != "Broadcast")
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "failed to find skiplayernorm subgraph";
            return false;
        }
        auto bias = broadcast->get_in_edge(0)->get_src();
        if (bias && bias->get_op_type() != "Constant")
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "failed to find skiplayernorm subgraph";
            return false;
        }

        bertfusion_group->fuse_group["inputs"] = {input, skip, bias};
        bertfusion_group->nodes_to_remove.insert({starting_node, add1, add2, broadcast});
    }

    bertfusion_group->helper_nodes.push_back(starting_node);
    bertfusion_group->edge_nodes.push_back(starting_node);

    return true;
}

bool SkipLayerNormFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 2 || inputs.size() == 3);
    auto input = inputs[0];
    auto skip = inputs[1];
    std::shared_ptr<GNode> bias = nullptr;
    if (inputs.size() == 3)
    {
        bias = inputs[2];
    }
    NNFUSION_CHECK(bertfusion_group->helper_nodes.size() == 1);
    auto layernorm = bertfusion_group->helper_nodes[0];
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // create skiplayernorm node
    NNFUSION_CHECK(layernorm->get_op_type() == "LayerNorm");
    auto layernorm_op = std::dynamic_pointer_cast<op::GenericOp>(layernorm->get_op_ptr());
    auto gamma = layernorm->get_in_edge(1)->get_src();
    auto beta = layernorm->get_in_edge(2)->get_src();

    auto& cfg = layernorm_op->localOpConfig.getRoot();
    float epsilon = cfg["epsilon"];

    nnfusion::op::OpConfig::any myConfig;
    myConfig["epsilon"] = epsilon;

    auto skiplayernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        "skip" + layernorm->get_name(), "SkipLayerNorm", myConfig);

    std::shared_ptr<GNode> skiplayernorm_gnode;
    if (bias != nullptr)
    {
        skiplayernorm_gnode = m_graph->add_node_and_edge(skiplayernorm_op,
                                                         {GNodeIndex{input, 0},
                                                          GNodeIndex{skip, 0},
                                                          GNodeIndex{gamma, 0},
                                                          GNodeIndex{beta, 0},
                                                          GNodeIndex{bias, 0}});
    }
    else
    {
        skiplayernorm_gnode = m_graph->add_node_and_edge(
            skiplayernorm_op,
            {GNodeIndex{input, 0}, GNodeIndex{skip, 0}, GNodeIndex{gamma, 0}, GNodeIndex{beta, 0}});
    }

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(skiplayernorm_gnode, 0, dst, y);
        }
    }

    return RemoveNodes(bertfusion_group->nodes_to_remove, skiplayernorm_gnode);
}
