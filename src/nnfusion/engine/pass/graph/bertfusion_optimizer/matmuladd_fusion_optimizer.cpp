// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "matmuladd_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool MatMulAddFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_op_type() != "Dot")
        return false;

    auto A_shape = node->get_input_shape(0);
    auto B_shape = node->get_input_shape(1);
    if (A_shape.size() != 2 || B_shape.size() != 2)
        return false;
    if (node->get_out_edges().size() != 1)
        return false;
    auto add = (*node->get_out_edges().begin())->get_dst();
    if (add->get_op_type() != "Add" ||
        add->get_output_element_type(0) != node->get_output_element_type(0))
        return false;
    return true;
}

bool MatMulAddFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                            std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto add = (*starting_node->get_out_edges().begin())->get_dst();
    auto matmul_a = starting_node->get_in_edge(0)->get_src();
    auto matmul_b = starting_node->get_in_edge(1)->get_src();

    std::shared_ptr<GNode> nodeC;
    for (auto in_edge : add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != starting_node)
        {
            nodeC = src;
            break;
        }
    }

    bertfusion_group->fuse_group["inputs"] = {matmul_a, matmul_b, nodeC};
    bertfusion_group->nodes_to_remove.insert({starting_node, add});
    bertfusion_group->edge_nodes.push_back(add);
    bertfusion_group->helper_nodes.push_back(starting_node);

    return true;
}

bool MatMulAddFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 3);
    auto matmul_a = inputs[0];
    auto matmul_b = inputs[1];
    auto nodeC = inputs[2];
    NNFUSION_CHECK(bertfusion_group->helper_nodes.size() == 1);
    auto matmul = bertfusion_group->helper_nodes[0];
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // create matmuladd node
    auto matmul_op = std::dynamic_pointer_cast<op::Dot>(matmul->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(matmul_op);
    bool trans_A = matmul_op->get_transpose_A();
    bool trans_B = matmul_op->get_transpose_B();
    nnfusion::op::OpConfig::any myConfig;
    myConfig["trans_A"] = trans_A;
    myConfig["trans_B"] = trans_B;

    auto matmuladd_op = std::make_shared<nnfusion::op::GenericOp>(
        matmul->get_name() + "add", "MatMulAdd", myConfig);
    auto matmuladd_gnode = m_graph->add_node_and_edge(
        matmuladd_op, {GNodeIndex{matmul_a, 0}, GNodeIndex{matmul_b, 0}, GNodeIndex{nodeC, 0}});

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(matmuladd_gnode, 0, dst, y);
        }
    }

    return RemoveNodes(bertfusion_group->nodes_to_remove, matmuladd_gnode);
}
