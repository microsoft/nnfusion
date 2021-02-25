// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attention_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool AttentionFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    // Starting Node has 4 children
    if (node->get_out_edges().size() != 4)
        return false;
    // Starting Node shape = [batch_size, sequence_length, hidden_size]
    auto shape = node->get_output_shape(0);
    if (shape.size() != 3)
        return false;
    auto hidden_size = shape[2];

    // Starting Node has 4 children:  3 of them are Dot
    auto edges = node->get_out_edges();
    if (edges.size() != 4)
        return false;
    int dot_count = 0;
    for (auto e : edges)
    {
        auto dst = e->get_dst();
        if (dst->get_op_type() == "Dot")
        {
            dot_count += 1;
        }
    }

    if (dot_count != 3)
        return false;

    return true;
}

bool AttentionFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                            std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    std::shared_ptr<GNode> ending_node;
    for (auto out_edge : starting_node->get_out_edges())
    {
        auto dst = out_edge->get_dst();
        if (dst->get_op_type() != "Dot")
        {
            ending_node = dst;
            break;
        }
    }
    NNFUSION_CHECK_NOT_NULLPTR(ending_node);

    //path_v
    std::vector<std::string> pattern_v1 = {ending_node->get_op_type(),
                                           "Add",
                                           "Dot",
                                           "Reshape",
                                           "Reshape",
                                           "Reshape",
                                           "BatchMatMul",
                                           "Broadcast",
                                           "Reshape",
                                           "Reshape",
                                           "Reshape",
                                           "Add",
                                           "Dot",
                                           starting_node->get_op_type()};

    std::vector<std::string> pattern_v2 = {ending_node->get_op_type(),
                                           "Dot",
                                           "Reshape",
                                           "Reshape",
                                           "Reshape",
                                           "BatchMatMul",
                                           "Broadcast",
                                           "Reshape",
                                           "Reshape",
                                           "Reshape",
                                           "Add",
                                           "Dot",
                                           starting_node->get_op_type()};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_v;
    if ((!FindPath(ending_node, pattern_v1, all_paths_v, true) &&
         !FindPath(ending_node, pattern_v2, all_paths_v, true)) ||
        all_paths_v.size() != 1 || all_paths_v[0][all_paths_v[0].size() - 1] != starting_node)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path v";
        return false;
    }
    size_t path_v_len = all_paths_v[0].size();
    std::shared_ptr<GNode> add = nullptr;
    // if (path_v_len > 13)
    //     add = all_paths_v[0][path_v_len - 13];
    // auto dot = all_paths_v[0][path_v_len - 12];
    auto reshape2 = all_paths_v[0][path_v_len - 11];
    // auto reshape1 = all_paths_v[0][path_v_len - 10];
    // auto reshape_after = all_paths_v[0][path_v_len - 9];
    auto qkv_batmatmul = all_paths_v[0][path_v_len - 8];
    // auto v_broadcast = all_paths_v[0][path_v_len - 7];
    // auto v_reshape_before = all_paths_v[0][path_v_len - 6];
    auto v_reshape2 = all_paths_v[0][path_v_len - 5];
    auto v_reshape1 = all_paths_v[0][path_v_len - 4];
    auto v_add = all_paths_v[0][path_v_len - 3];
    auto v_dot = all_paths_v[0][path_v_len - 2];

    auto v_reshape1_shape = v_reshape1->get_output_shape(0);

    if (v_add->get_out_edges().size() != 1 || v_dot->get_out_edges().size() != 1 ||
        v_reshape2->get_out_edges().size() != 1 || qkv_batmatmul->get_out_edges().size() != 1 ||
        v_reshape2->get_out_edges().size() != 1 || v_reshape1->get_out_edges().size() != 1 ||
        v_reshape1_shape.size() != 4)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path v";
        return false;
    }
    size_t num_heads = v_reshape1_shape[2];
    size_t head_size = v_reshape1_shape[3];
    size_t hidden_size = starting_node->get_output_shape(0)[2];
    if (hidden_size != num_heads * head_size)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path v";
        return false;
    }

    if (path_v_len > 13)
    {
        bertfusion_group->nodes_to_remove.insert(all_paths_v[0].begin() + 3,
                                                 all_paths_v[0].end() - 1);
    }
    else
    {
        bertfusion_group->nodes_to_remove.insert(all_paths_v[0].begin() + 2,
                                                 all_paths_v[0].end() - 1);
    }

    //path_mask
    std::vector<std::string> pattern_mask1 = {qkv_batmatmul->get_op_type(),
                                              "Broadcast",
                                              "Reshape",
                                              "Softmax",
                                              "Add",
                                              "Broadcast",
                                              "Reshape",
                                              "Multiply",
                                              "Subtract",
                                              "Convert",
                                              "Reshape",
                                              "Reshape"};
    std::vector<std::string> pattern_mask2 = {qkv_batmatmul->get_op_type(),
                                              "Broadcast",
                                              "Reshape",
                                              "Softmax",
                                              "Add",
                                              "Broadcast",
                                              "Reshape",
                                              "Multiply",
                                              "Subtract",
                                              "Reshape",
                                              "Reshape"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_mask;
    if ((!FindPath(qkv_batmatmul, pattern_mask1, all_paths_mask, true) &&
         !FindPath(qkv_batmatmul, pattern_mask2, all_paths_mask, true)) ||
        all_paths_mask.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path mask";
        return false;
    }

    // auto mask_broadcast = all_paths_mask[0][1];
    // auto mask_reshape_before = all_paths_mask[0][2];
    // auto mask_softmax = all_paths_mask[0][3];
    auto mask_add = all_paths_mask[0][4];
    auto mask_broadcast_for_add = all_paths_mask[0][5];
    // auto mask_broadcast_reshape = all_paths_mask[0][6];
    // auto mask_mul = all_paths_mask[0][7];
    auto mask_sub = all_paths_mask[0][8];
    // auto mask_reshape2 = all_paths_mask[0][all_paths_mask[0].size() - 2];
    auto mask_reshape1 = all_paths_mask[0][all_paths_mask[0].size() - 1];
    std::shared_ptr<GNode> mask_convert = nullptr;
    if (all_paths_mask[0].size() == pattern_mask1.size())
    {
        mask_convert = all_paths_mask[0][all_paths_mask[0].size() - 3];
        NNFUSION_CHECK(mask_convert != mask_sub);
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_mask[0].begin() + 1,
                                             all_paths_mask[0].begin() + 5);
    if (mask_broadcast_for_add->get_out_edges().size() == 1)
        bertfusion_group->nodes_to_remove.insert(all_paths_mask[0].begin() + 5,
                                                 all_paths_mask[0].end());

    //path_q and path_k
    std::vector<std::string> pattern_qk = {mask_add->get_op_type(),
                                           "Divide",
                                           "Reshape",
                                           "BatchMatMul",
                                           "Broadcast",
                                           "Reshape",
                                           "Reshape",
                                           "Reshape",
                                           "Add",
                                           "Dot",
                                           starting_node->get_op_type()};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_qk;
    if (!FindPath(mask_add, pattern_qk, all_paths_qk, true) || all_paths_qk.size() != 2 ||
        all_paths_qk[0][pattern_qk.size() - 1] != starting_node ||
        all_paths_qk[1][pattern_qk.size() - 1] != starting_node)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q and k";
        return false;
    }

    if (all_paths_qk[0][1] != all_paths_qk[0][1] || all_paths_qk[0][2] != all_paths_qk[0][2])
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q and k";
        return false;
    }
    auto qk_div = all_paths_qk[0][1];
    auto qk_reshape = all_paths_qk[0][2];
    auto qk_batmatmul = all_paths_qk[0][3];
    size_t q_idx, k_idx;
    if (all_paths_qk[0][4] == qk_batmatmul->get_in_edge(0)->get_src())
    {
        q_idx = 0;
        k_idx = 1;
    }
    else
    {
        q_idx = 1;
        k_idx = 0;
    }

    // auto q_broadcast = all_paths_qk[q_idx][4];
    // auto q_reshape_before = all_paths_qk[q_idx][5];
    // auto q_reshape2 = all_paths_qk[q_idx][6];
    auto q_reshape1 = all_paths_qk[q_idx][7];
    auto q_add = all_paths_qk[q_idx][8];
    auto q_dot = all_paths_qk[q_idx][9];
    // auto k_broadcast = all_paths_qk[k_idx][4];
    // auto k_reshape_before = all_paths_qk[k_idx][5];
    // auto k_reshape2 = all_paths_qk[k_idx][6];
    // auto k_reshape1 = all_paths_qk[k_idx][7];
    auto k_add = all_paths_qk[k_idx][8];
    auto k_dot = all_paths_qk[k_idx][9];

    auto q_reshape1_shape = q_reshape1->get_output_shape(0);
    if (q_reshape1_shape.size() != 4 || q_reshape1_shape[2] != num_heads ||
        q_reshape1_shape[3] != head_size)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q and k";
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_qk[0].begin() + 1,
                                             all_paths_qk[0].end() - 1);
    bertfusion_group->nodes_to_remove.insert(all_paths_qk[1].begin() + 1,
                                             all_paths_qk[1].end() - 1);

    bertfusion_group->fuse_group["mask"] = {mask_reshape1};
    bertfusion_group->fuse_group["qkv_weight"] = {q_dot, k_dot, v_dot};
    bertfusion_group->fuse_group["qkv_bias"] = {q_add, k_add, v_add};
    bertfusion_group->fuse_group["inputs"] = {starting_node};
    bertfusion_group->helper_nodes.push_back(v_reshape1);
    bertfusion_group->edge_nodes.push_back(reshape2);

    return true;
}

bool AttentionFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("mask") != fuse_group.end());
    NNFUSION_CHECK(fuse_group.find("qkv_weight") != fuse_group.end());
    NNFUSION_CHECK(fuse_group.find("qkv_bias") != fuse_group.end());
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& mask = fuse_group["mask"];
    auto& qkv_weight = fuse_group["qkv_weight"];
    auto& qkv_bias = fuse_group["qkv_bias"];
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(mask.size() == 1);
    NNFUSION_CHECK(qkv_weight.size() == 3);
    NNFUSION_CHECK(qkv_bias.size() == 3);
    NNFUSION_CHECK(inputs.size() == 1);
    auto mask_reshape1 = mask[0];
    auto q_dot = qkv_weight[0];
    auto k_dot = qkv_weight[1];
    auto v_dot = qkv_weight[2];
    auto q_add = qkv_bias[0];
    auto k_add = qkv_bias[1];
    auto v_add = qkv_bias[2];
    auto input_node = inputs[0];
    NNFUSION_CHECK(bertfusion_group->helper_nodes.size() == 1);
    auto v_reshape1 = bertfusion_group->helper_nodes[0];
    auto v_reshape1_shape = v_reshape1->get_output_shape(0);
    size_t num_heads = v_reshape1_shape[2];
    size_t head_size = v_reshape1_shape[3];
    auto input_shape = input_node->get_output_shape(0);
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // now we will begin to fuse subgraph
    //1. create mask_index
    auto mask_input = mask_reshape1->get_in_edge(0)->get_src();
    auto mask_index = GetorCreateMaskIndex(mask_input);
    if (mask_index == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to create mask index";
        return false;
    }

    // 2. merge qkv weight
    auto q_weight = q_dot->get_in_edge(1)->get_src();
    auto k_weight = k_dot->get_in_edge(1)->get_src();
    auto v_weight = v_dot->get_in_edge(1)->get_src();
    size_t hidden_size = input_node->get_output_shape(0)[2];
    auto qkv_weight_node = MergeQkvWeights(q_weight, k_weight, v_weight, hidden_size, true);
    if (qkv_weight_node == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to merge qkv weights";
        return false;
    }
    std::vector<std::shared_ptr<GNode>> qkv_weights{q_weight, k_weight, v_weight};
    for (auto weight : qkv_weights)
    {
        if (weight->get_out_edges().size() == 1)
        {
            bertfusion_group->nodes_to_remove.insert(weight);
        }
    }
    // bertfusion_group->nodes_to_remove.insert({q_weight, k_weight, v_weight});

    // 3. merge qkv bias
    auto q_bias_broadcast = q_add->get_in_edge(1)->get_src();
    auto q_bias = q_bias_broadcast->get_in_edge(0)->get_src();
    auto k_bias_broadcast = k_add->get_in_edge(1)->get_src();
    auto k_bias = k_bias_broadcast->get_in_edge(0)->get_src();
    auto v_bias_broadcast = v_add->get_in_edge(1)->get_src();
    auto v_bias = v_bias_broadcast->get_in_edge(0)->get_src();
    auto qkv_bias_node = MergeQkvWeights(q_bias, k_bias, v_bias, hidden_size, false);

    if (qkv_bias_node == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to merge qkv bias";
        return false;
    }

    std::vector<std::shared_ptr<GNode>> qkv_bias_broadcast{
        q_bias_broadcast, k_bias_broadcast, k_bias_broadcast};

    for (auto bias : qkv_bias_broadcast)
    {
        if (bias->get_out_edges().size() == 1)
        {
            bertfusion_group->nodes_to_remove.insert(bias);
        }
    }

    // bertfusion_group->nodes_to_remove.insert(
    //     {q_bias, k_bias, v_bias, q_bias_broadcast, k_bias_broadcast, k_bias_broadcast});

    // 4. create broadcast node
    auto broadcasted_op = std::make_shared<op::Broadcast>(
        nnfusion::Shape({batch_size * sequence_length, 3 * hidden_size}), nnfusion::AxisSet({0}));
    broadcasted_op->set_name(qkv_bias_node->get_name() + "_broadcast");
    auto broadcasted_gnode =
        m_graph->add_node_and_edge(broadcasted_op, {GNodeIndex{qkv_bias_node, 0}});

    // 5. create reshape node
    nnfusion::AxisVector ng_axis_order(input_shape.size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(
        ng_axis_order, nnfusion::Shape({batch_size * sequence_length, hidden_size}));
    reshape_op->set_name(input_node->get_name() + "_reshape");
    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {GNodeIndex{input_node, 0}});

    // 6. create dot node
    auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
    dot_op->set_name(input_node->get_name() + "_dot");
    auto dot_gnode = m_graph->add_node_and_edge(
        dot_op, {GNodeIndex{reshape_gnode, 0}, GNodeIndex{qkv_weight_node, 0}});

    // 7. create add node
    auto add_op = std::make_shared<op::Add>();
    add_op->set_name(input_node->get_name() + "_add");
    auto add_gnode = m_graph->add_node_and_edge(
        add_op, {GNodeIndex{dot_gnode, 0}, GNodeIndex{broadcasted_gnode, 0}});

    // 8. create attention node
    nnfusion::op::OpConfig::any myConfig;
    myConfig["num_heads"] = num_heads;
    myConfig["batch_size"] = batch_size;
    myConfig["sequence_length"] = sequence_length;
    myConfig["head_size"] = head_size;
    // myConfig["unidirectional"] = false;
    // myConfig["past_sequence_length"] = 0;
    auto attention_op = std::make_shared<nnfusion::op::GenericOp>(
        input_node->get_name() + "_attention", "Attention", myConfig);
    auto attention_gnode = m_graph->add_node_and_edge(
        attention_op, {GNodeIndex{add_gnode, 0}, GNodeIndex{mask_index, 0}});

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(attention_gnode, 0, dst, y);
        }
    }

    if (!RemoveNodes(bertfusion_group->nodes_to_remove, attention_gnode))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "remove nodes failed.";
    }
    std::unordered_set<std::shared_ptr<GNode>> remove_at_last;
    if (q_bias->get_out_edges().size() == 0)
    {
        remove_at_last.insert(q_bias);
    }
    if (k_bias->get_out_edges().size() == 0)
    {
        remove_at_last.insert(k_bias);
    }
    if (v_bias->get_out_edges().size() == 0)
    {
        remove_at_last.insert(v_bias);
    }
    return RemoveNodes(remove_at_last, qkv_bias_node);
}

std::shared_ptr<GNode> AttentionFusionOptimizer::MergeQkvWeights(std::shared_ptr<GNode> q_weight,
                                                                 std::shared_ptr<GNode> k_weight,
                                                                 std::shared_ptr<GNode> v_weight,
                                                                 size_t hidden_size,
                                                                 bool is_matmul)
{
    // Lookup in map, and return the mask index if created.
    std::string name = q_weight->get_name() + k_weight->get_name() + v_weight->get_name();
    auto search = qkv_weight_map.find(name);
    if (search != qkv_weight_map.end())
    {
        return search->second;
    }
    auto q_weight_op = std::dynamic_pointer_cast<op::Constant>(q_weight->get_op_ptr());
    auto k_weight_op = std::dynamic_pointer_cast<op::Constant>(k_weight->get_op_ptr());
    auto v_weight_op = std::dynamic_pointer_cast<op::Constant>(v_weight->get_op_ptr());

    // NNFUSION_CHECK_NOT_NULLPTR(q_weight_op) << q_weight->get_op_type();
    // NNFUSION_CHECK_NOT_NULLPTR(k_weight_op) << k_weight->get_op_type();
    // NNFUSION_CHECK_NOT_NULLPTR(v_weight_op) << v_weight->get_op_type();

    if (!q_weight_op || !k_weight_op || !v_weight_op)
    {
        return nullptr;
    }
    const char* q_weight_dptr = (char*)q_weight_op->get_data_ptr();
    const char* k_weight_dptr = (char*)k_weight_op->get_data_ptr();
    const char* v_weight_dptr = (char*)v_weight_op->get_data_ptr();
    NNFUSION_CHECK_NOT_NULLPTR(q_weight_dptr);
    NNFUSION_CHECK_NOT_NULLPTR(k_weight_dptr);
    NNFUSION_CHECK_NOT_NULLPTR(v_weight_dptr);

    auto weight_dtype = q_weight_op->get_type();

    std::vector<char> qkv_weight_data;
    size_t step = hidden_size * weight_dtype.size();
    nnfusion::Shape qkv_weight_shape;

    if (is_matmul)
    {
        MergeMatMulWeights(q_weight_dptr, k_weight_dptr, v_weight_dptr, step, qkv_weight_data);
        qkv_weight_shape = {hidden_size, 3 * hidden_size};
    }
    else
    {
        MergeWeights(q_weight_dptr, k_weight_dptr, v_weight_dptr, step, qkv_weight_data);
        qkv_weight_shape = {3 * hidden_size};
    }

    std::shared_ptr<op::Constant> new_qkv_weight_op =
        std::make_shared<op::Constant>(weight_dtype, qkv_weight_shape, qkv_weight_data.data());
    new_qkv_weight_op->set_name("mergedqkv" + q_weight_op->get_name());
    std::shared_ptr<GNode> new_qkv_weight_gnode =
        m_graph->add_node_and_edge(new_qkv_weight_op, GNodeVector());
    qkv_weight_map.insert(
        std::pair<std::string, std::shared_ptr<GNode>>(name, new_qkv_weight_gnode));
    return new_qkv_weight_gnode;
}

void AttentionFusionOptimizer::MergeWeights(const char* q_weight_dptr,
                                            const char* k_weight_dptr,
                                            const char* v_weight_dptr,
                                            size_t step,
                                            std::vector<char>& qkv_weight_data)
{
    for (size_t i = 0; i < step; i++)
    {
        qkv_weight_data.push_back(*q_weight_dptr);
        q_weight_dptr++;
    }

    for (size_t i = 0; i < step; i++)
    {
        qkv_weight_data.push_back(*k_weight_dptr);
        k_weight_dptr++;
    }

    for (size_t i = 0; i < step; i++)
    {
        qkv_weight_data.push_back(*v_weight_dptr);
        v_weight_dptr++;
    }
}

void AttentionFusionOptimizer::MergeMatMulWeights(const char* q_weight_dptr,
                                                  const char* k_weight_dptr,
                                                  const char* v_weight_dptr,
                                                  size_t step,
                                                  std::vector<char>& qkv_weight_data)
{
    for (size_t i = 0; i < step;
         i++, q_weight_dptr += step, k_weight_dptr += step, v_weight_dptr += step)
    {
        MergeWeights(q_weight_dptr, k_weight_dptr, v_weight_dptr, step, qkv_weight_data);
    }
}

std::shared_ptr<GNode>
    AttentionFusionOptimizer::GetorCreateMaskIndex(std::shared_ptr<GNode> mask_input)
{
    // Lookup in map, and return the mask index if created.
    auto search = mask_index_map.find(mask_input->get_name());
    if (search != mask_index_map.end())
    {
        return search->second;
    }

    auto mask_shape = mask_input->get_output_shape(0);
    if (mask_shape.size() != 2)
        return nullptr;

    std::shared_ptr<GNode> mask_index = mask_input;
    auto dtype = mask_input->get_output_element_type(0);
    if (dtype != element::i32)
    {
        auto cast_op = std::make_shared<op::Convert>(element::i32);
        cast_op->set_name(mask_input->get_name() + "_to_int32");
        std::shared_ptr<GNode> cast_gnode = m_graph->add_node_and_edge(cast_op, {mask_input});
        mask_index = cast_gnode;
    }

    mask_index_map.insert(
        std::pair<std::string, std::shared_ptr<GNode>>(mask_input->get_name(), mask_index));
    return mask_index;
}
