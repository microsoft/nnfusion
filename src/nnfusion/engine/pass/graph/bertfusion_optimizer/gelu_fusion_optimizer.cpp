// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gelu_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool GeluFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_out_edges().size() != 2)
        return false;
    std::shared_ptr<GNode> divide;
    int mul_count = 0;
    int div_count = 0;
    for (auto edge : node->get_out_edges())
    {
        auto dst = edge->get_dst();
        if (dst->get_op_type() == "Multiply")
        {
            mul_count += 1;
        }
        else if (dst->get_op_type() == "Divide")
        {
            div_count += 1;
            divide = dst;
        }
    }

    if (mul_count != 1 || div_count != 1)
        return false;

    return true;
}

/*
     This function fuses subgraph like the following into one Gelu node.
     Subgraph pattern 1:
                   +-------Mul(0.5)---------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul ==>
                          (B=1.4142...)        (1)

      Subgraph pattern 2:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)

       After Fusion:
                [root]--> Gelu ==>
*/
bool GeluFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                       std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    std::shared_ptr<GNode> multiply1, divide;
    for (auto out_edge : starting_node->get_out_edges())
    {
        auto dst = out_edge->get_dst();
        if (dst->get_op_type() == "Multiply")
            multiply1 = dst;
        else if (dst->get_op_type() == "Divide")
            divide = dst;
    }

    auto broadcast_before_div = divide->get_in_edge(1)->get_src();
    if (broadcast_before_div->get_op_type() != "Broadcast")
    {
        // NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2)";
        return false;
    }
    auto const_before_div = broadcast_before_div->get_in_edge(0)->get_src();
    std::vector<float> const_value;
    bool status = nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_div, &const_value);
    if (!status || const_value.size() != 1)
    {
        // NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2)";
        return false;
    }

    float approximate_sqrt2 = 1.4142099618911743f;
    float diff = std::abs(const_value[0] - approximate_sqrt2);
    const float atol = 1e-8f;
    const float rtol = 1e-3f;
    if (diff > (atol + rtol * std::abs(approximate_sqrt2)))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2), got " << const_value[0];
        return false;
    }

    // find gelu subgraph
    std::vector<std::string> pattern = {"Divide", "Erf", "Add", "Multiply"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths;
    if (!FindPath(divide, pattern, all_paths, false) || all_paths.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph.";
    }

    auto erf = all_paths[0][1];
    auto add = all_paths[0][2];
    std::shared_ptr<GNode> broadcast_before_add, const_before_add;
    for (auto in_edge : add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != erf)
        {
            broadcast_before_add = src;
            if (broadcast_before_add->get_op_type() != "Broadcast")
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                return false;
            }
            const_before_add = broadcast_before_add->get_in_edge(0)->get_src();
            const_value.clear();
            status =
                nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_add, &const_value);
            if (!status || const_value.size() != 1 || const_value[0] != 1)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                return false;
            }
            break;
        }
    }

    std::shared_ptr<GNode> multiply2, broadcast_before_mul, const_before_mul;
    if (all_paths[0].back() == multiply1) // pattern 2
    {
        for (auto out_edge : all_paths[0].back()->get_out_edges())
        {
            auto dst = out_edge->get_dst();
            if (dst->get_op_type() == "Multiply")
            {
                multiply2 = dst;
                for (auto in_edge : multiply2->get_in_edges())
                {
                    auto src = in_edge->get_src();
                    if (src != multiply1)
                    {
                        broadcast_before_mul = src;
                        if (broadcast_before_mul->get_op_type() != "Broadcast")
                        {
                            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                            return false;
                        }
                        const_before_mul = broadcast_before_mul->get_in_edge(0)->get_src();
                        const_value.clear();
                        status = nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_mul,
                                                                                 &const_value);
                        if (!status || const_value.size() != 1 || const_value[0] != 0.5)
                        {
                            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                            return false;
                        }
                        break;
                    }
                }
                break;
            }
        }
    }
    else // pattern 1
    {
        multiply2 = all_paths[0].back();
        for (auto in_edge : multiply1->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src != starting_node)
            {
                broadcast_before_mul = src;
                if (broadcast_before_mul->get_op_type() != "Broadcast")
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                    return false;
                }
                const_before_mul = broadcast_before_mul->get_in_edge(0)->get_src();
                const_value.clear();
                status =
                    nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_mul, &const_value);
                if (!status || const_value.size() != 1 || const_value[0] != 0.5)
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                    return false;
                }
                break;
            }
        }
    }

    if (multiply2 == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
        return false;
    }

    bertfusion_group->fuse_group["inputs"] = {starting_node};
    bertfusion_group->nodes_to_remove.insert(all_paths[0].begin(), all_paths[0].end());
    bertfusion_group->nodes_to_remove.insert({multiply1,
                                              multiply2,
                                              broadcast_before_div,
                                              const_before_div,
                                              broadcast_before_add,
                                              const_before_add,
                                              broadcast_before_mul,
                                              const_before_mul});
    bertfusion_group->edge_nodes.push_back(multiply2);

    return true;
}

bool GeluFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 1);
    auto input = inputs[0];
    NNFUSION_CHECK(bertfusion_group->helper_nodes.empty());
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // create gelu node
    auto gelu_op = std::make_shared<nnfusion::op::Gelu>();

    auto gelu_gnode = m_graph->add_node_and_edge(gelu_op, {GNodeIndex{input, 0}});

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(gelu_gnode, 0, dst, y);
        }
    }

    return RemoveNodes(bertfusion_group->nodes_to_remove, gelu_gnode);
}
