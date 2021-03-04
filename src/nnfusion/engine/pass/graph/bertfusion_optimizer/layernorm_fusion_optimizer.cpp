// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "layernorm_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool LayerNormFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_out_edges().size() != 2)
        return false;

    int sum_count = 0;
    int sub_count = 0;
    for (auto out_edge : node->get_out_edges())
    {
        auto dst = out_edge->get_dst();
        if (dst->get_op_type() == "Sum")
        {
            sum_count += 1;
        }
        else if (dst->get_op_type() == "Subtract")
        {
            sub_count += 1;
        }
    }

    if (sum_count != 1 || sub_count != 1)
        return false;

    return true;
}

bool LayerNormFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                            std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    std::shared_ptr<GNode> ending_node;
    for (auto out_edge : starting_node->get_out_edges())
    {
        auto dst = out_edge->get_dst();
        if (dst->get_op_type() == "Subtract")
        {
            ending_node = dst;
            break;
        }
    }
    NNFUSION_CHECK_NOT_NULLPTR(ending_node);

    // find reducemean1
    std::vector<std::vector<std::shared_ptr<GNode>>> reducemean1_mainpath;
    if (!FindReduceMean(starting_node, bertfusion_group, reducemean1_mainpath))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find the first reducemean subgraph";
        return false;
    }
    auto reducemean1 = reducemean1_mainpath[0].back();

    // find path to reducemean2
    std::vector<std::string> pattern_to_reducemean2_1 = {
        reducemean1->get_op_type(), "Reshape", "Broadcast", "Subtract", "Power"};
    std::vector<std::string> pattern_to_reducemean2_2 = {
        reducemean1->get_op_type(), "Reshape", "Broadcast", "Subtract", "Convert", "Power"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_to_reducemean2;
    if ((!FindPath(reducemean1, pattern_to_reducemean2_1, all_paths_to_reducemean2, false) &&
         !FindPath(reducemean1, pattern_to_reducemean2_2, all_paths_to_reducemean2, false)) ||
        all_paths_to_reducemean2.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to the second reducemean subgraph";
        return false;
    }

    auto subtract = all_paths_to_reducemean2[0][3];
    if (subtract != ending_node)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to the second reducemean subgraph";
        return false;
    }

    auto power = all_paths_to_reducemean2[0].back();
    // auto power_broadcast = power->get_in_edge(1)->get_src();
    // if (power_broadcast->get_in_edges().empty() ||
    //     !power_broadcast->get_in_edge(0)->get_src()->is_constant())
    // {
    //     NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to the second reducemean subgraph";
    //     return false;
    // }

    // auto power_const = power_broadcast->get_in_edge(0)->get_src();
    bertfusion_group->nodes_to_remove.insert(all_paths_to_reducemean2[0].begin(),
                                             all_paths_to_reducemean2[0].end());
    // bertfusion_group->nodes_to_remove.insert({power, power_broadcast, power_const});

    // find reducemean2
    std::vector<std::vector<std::shared_ptr<GNode>>> reducemean2_mainpath;
    if (!FindReduceMean(power, bertfusion_group, reducemean2_mainpath))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find the first reducemean subgraph";
        return false;
    }

    auto reducemean2 = reducemean2_mainpath[0].back();

    // find path to the last add
    std::vector<std::string> pattern_to_last_add = {reducemean2->get_op_type(),
                                                    "Add",
                                                    "Sqrt",
                                                    "Reshape",
                                                    "Broadcast",
                                                    "Divide",
                                                    "Multiply",
                                                    "Add"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_to_last_add;
    if (!FindPath(reducemean2, pattern_to_last_add, all_paths_to_last_add, false) ||
        all_paths_to_last_add.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to  to the last add";
        return false;
    }

    auto divide = all_paths_to_last_add[0][5];
    if (divide->get_in_edge(0)->get_src() != subtract)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to the last add";
        return false;
    }

    auto add2 = all_paths_to_last_add[0][1];
    auto multiply = all_paths_to_last_add[0][6];
    auto last_add = all_paths_to_last_add[0].back();

    std::shared_ptr<GNode> epsilon, scale, bias, epsilon_broadcast, epsilon_reshape,
        scale_broadcast, bias_broadcast;
    for (auto in_edge : add2->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != reducemean2 && src->get_op_type() == "Broadcast")
        {
            epsilon_broadcast = src;
            epsilon_reshape = src->get_in_edge(0)->get_src();
            if (epsilon_reshape->get_op_type() != "Reshape")
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
                return false;
            }
            epsilon = epsilon_reshape->get_in_edge(0)->get_src();
        }
    }
    for (auto in_edge : multiply->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != divide && src->get_op_type() == "Broadcast")
        {
            scale_broadcast = src;
            scale = src->get_in_edge(0)->get_src();
        }
    }

    for (auto in_edge : last_add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != multiply && src->get_op_type() == "Broadcast")
        {
            bias_broadcast = src;
            bias = src->get_in_edge(0)->get_src();
        }
    }

    if (scale == nullptr || bias == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to  to the last add";
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_to_last_add[0].begin(),
                                             all_paths_to_last_add[0].end());
    bertfusion_group->nodes_to_remove.insert(
        {epsilon_broadcast, scale_broadcast, bias_broadcast, epsilon_reshape, epsilon});

    bertfusion_group->fuse_group["inputs"] = {starting_node, scale, bias};
    bertfusion_group->helper_nodes.push_back(epsilon);
    auto reducemean1_sum = reducemean1_mainpath[0][1];
    bertfusion_group->helper_nodes.push_back(reducemean1_sum);
    bertfusion_group->edge_nodes.push_back(last_add);

    return true;
}

bool LayerNormFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 3);
    auto input_node = inputs[0];
    auto scale = inputs[1];
    auto bias = inputs[2];
    NNFUSION_CHECK(bertfusion_group->helper_nodes.size() == 2);
    auto epsilon = bertfusion_group->helper_nodes[0];
    auto reducemean1_sum = bertfusion_group->helper_nodes[1];
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // create layernorm node
    auto reducemean1_op = std::dynamic_pointer_cast<op::Sum>(reducemean1_sum->get_op_ptr());
    std::vector<size_t> axis_vec;
    for (auto axis : reducemean1_op->get_reduction_axes())
    {
        axis_vec.push_back(axis);
    }

    if (axis_vec.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find axis";
        return false;
    }

    std::vector<float> epsilon_value;
    bool status = nnfusion::frontend::GetValueFromNGraphOp<float>(epsilon, &epsilon_value);
    if (!status || epsilon_value.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
        return false;
    }

    nnfusion::op::OpConfig::any myConfig;
    myConfig["axis"] = axis_vec[0];
    myConfig["epsilon"] = epsilon_value[0];
    auto layernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        input_node->get_name() + "_layernorm", "LayerNorm", myConfig);
    auto layernorm_gnode = m_graph->add_node_and_edge(
        layernorm_op, {GNodeIndex{input_node, 0}, GNodeIndex{scale, 0}, GNodeIndex{bias, 0}});

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(layernorm_gnode, 0, dst, y);
        }
    }

    return RemoveNodes(bertfusion_group->nodes_to_remove, layernorm_gnode);
}

bool LayerNormFusionOptimizer::FindReduceMean(
    std::shared_ptr<GNode> node,
    std::shared_ptr<BertFusionGroup> bertfusion_group,
    std::vector<std::vector<std::shared_ptr<GNode>>>& main_path)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    std::vector<std::string> pattern_main = {node->get_op_type(), "Sum", "Divide", "Reshape"};
    if (!FindPath(node, pattern_main, main_path, false) || main_path.empty())
    {
        return false;
    }

    if (main_path.size() != 1)
    {
        return false;
    }

    auto divide = main_path[0][2];
    std::vector<std::string> pattern_fork = {"Divide", "Broadcast", "Constant"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_fork;
    if (!FindPath(divide, pattern_fork, all_paths_fork, true) || all_paths_fork.size() != 1)
    {
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(main_path[0].begin() + 1, main_path[0].end());
    bertfusion_group->nodes_to_remove.insert(all_paths_fork[0].begin(), all_paths_fork[0].end());
    return true;
}
