// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attention_fusion_optimizer.hpp"
#include <queue>
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/op.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool AttentionFusionOptimizer::Optimize()
{
    for (auto node : m_graph->get_ordered_ops())
    {
        // Starting Node shape = [batch_size, sequence_length, hidden_size]
        auto shape = node->get_output_shape(0);
        if (shape.size() != 3)
            continue;
        auto hidden_size = shape[2];

        // Starting Node has 4 children: 1 Add, 3 MatMul
        auto edges = node->get_out_edges();
        if (edges.size() != 4)
            continue;
        int add_count = 0;
        int matmul_count = 0;
        std::shared_ptr<GNode> add_node;
        for (auto e : edges)
        {
            auto dst = e->get_dst();
            if (dst->get_op_type() == "Add")
            {
                add_count += 1;
                add_node = dst;
            }
            else if (dst->get_op_type() == "Dot")
            {
                matmul_count += 1;
            }
        }

        if (add_count != 1 || matmul_count != 3)
            continue;

        FuseSubGraph(node, add_node, hidden_size);
    }

    return true;
}

bool AttentionFusionOptimizer::FuseSubGraph(std::shared_ptr<GNode> starting_node,
                                            std::shared_ptr<GNode> ending_node,
                                            size_t hidden_size)
{
    //path_v
    std::vector<std::string> pattern_v = {"Add",
                                          "Add",
                                          "Dot",
                                          "Reshape",
                                          "Reshape",
                                          "Dot",
                                          "Reshape",
                                          "Reshape",
                                          "Add",
                                          "Dot",
                                          starting_node->get_op_type()};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_v;
    if (!ReverseFindPath(ending_node, pattern_v, all_paths_v) || all_paths_v.size() != 1 ||
        all_paths_v[0][pattern_v.size() - 1] != starting_node)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path v";
        return false;
    }

    auto add = all_paths_v[0][1];
    auto matmul = all_paths_v[0][2];
    auto reshape = all_paths_v[0][3];
    auto transpose = all_paths_v[0][4];
    auto qkv_matmul = all_paths_v[0][5];
    auto v_transpose = all_paths_v[0][6];
    auto v_reshape = all_paths_v[0][7];
    auto v_add = all_paths_v[0][8];
    auto v_matmul = all_paths_v[0][9];
    auto v_reshape_shape = v_reshape->get_output_shape(0);

    if (v_add->get_out_edges().size() != 1 || v_matmul->get_out_edges().size() != 1 ||
        transpose->get_out_edges().size() != 1 || qkv_matmul->get_out_edges().size() != 1 ||
        v_transpose->get_out_edges().size() != 1 || v_reshape->get_out_edges().size() != 1 ||
        v_reshape_shape.size() != 4)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path v";
        return false;
    }
    size_t num_heads = v_reshape_shape[2];
    size_t head_size = v_reshape_shape[3];
    if (hidden_size != num_heads * head_size)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path v";
        return false;
    }

    //path_mask
    std::vector<std::string> pattern_mask1 = {
        "Dot", "Softmax", "Add", "Multiply", "Subtract", "Reshape", "Reshape"};
    std::vector<std::string> pattern_mask2 = {
        "Dot", "Softmax", "Add", "Multiply", "Subtract", "Convert", "Reshape", "Reshape"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_mask;
    if ((!ReverseFindPath(qkv_matmul, pattern_mask1, all_paths_mask) &&
         !ReverseFindPath(qkv_matmul, pattern_mask2, all_paths_mask)) ||
        all_paths_mask.size() != 1)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path mask";
        return false;
    }
    auto mask_softmax = all_paths_mask[0][1];
    auto mask_add = all_paths_mask[0][2];
    auto mask_mul = all_paths_mask[0][3];
    auto mask_sub = all_paths_mask[0][4];
    auto mask_unsqueeze2 = all_paths_mask[0][all_paths_mask.size() - 2];
    auto mask_unsqueeze1 = all_paths_mask[0][all_paths_mask.size() - 1];
    std::shared_ptr<GNode> mask_cast = nullptr;
    if (all_paths_mask[0].size() == 8)
        mask_cast = all_paths_mask[0][5];

    //path_q and path_k
    std::vector<std::string> pattern_qk = {
        "Add", "Divide", "Dot", "Reshape", "Reshape", "Add", "Dot", starting_node->get_op_type()};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_qk;
    if (!ReverseFindPath(mask_add, pattern_qk, all_paths_qk) || all_paths_qk.size() != 2 ||
        all_paths_qk[0][pattern_qk.size() - 1] != starting_node ||
        all_paths_qk[1][pattern_qk.size() - 1] != starting_node)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path q and k";
        return false;
    }

    if (all_paths_qk[0][1] != all_paths_qk[0][1] || all_paths_qk[0][2] != all_paths_qk[0][2])
    {
        NNFUSION_LOG(ERROR) << "Failed to find path q and k";
        return false;
    }
    auto qk_div = all_paths_qk[0][1];
    auto qk_matmul = all_paths_qk[0][2];

    if (qk_matmul->get_in_edges().size() != 2)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path q and k";
        return false;
    }

    size_t q_idx, k_idx;
    if (all_paths_qk[0][3] == qk_matmul->get_in_edge(0)->get_src())
    {
        q_idx = 0;
        k_idx = 1;
    }
    else
    {
        q_idx = 1;
        k_idx = 0;
    }

    auto q_transpose = all_paths_qk[q_idx][3];
    auto q_reshape = all_paths_qk[q_idx][4];
    auto q_add = all_paths_qk[q_idx][5];
    auto q_matmul = all_paths_qk[q_idx][6];
    auto k_transpose = all_paths_qk[k_idx][3];
    auto k_reshape = all_paths_qk[k_idx][4];
    auto k_add = all_paths_qk[k_idx][5];
    auto k_matmul = all_paths_qk[k_idx][6];
    auto q_reshape_shape = q_reshape->get_output_shape(0);
    if (q_reshape_shape.size() != 4 || q_reshape_shape[2] != num_heads ||
        q_reshape_shape[3] != head_size)
    {
        NNFUSION_LOG(ERROR) << "Failed to find path q and k";
        return false;
    }

    // now we will begin to fuse subgraph
    //1. create mask_index
    auto mask_input = mask_unsqueeze1->get_in_edge(0)->get_src();
    auto mask_index = GetorCreateMaskIndex(mask_input);
    if (mask_index == nullptr)
    {
        NNFUSION_LOG(ERROR) << "Failed to create mask index";
        return false;
    }

    // 2. merge qkv weight
    auto q_weight = q_matmul->get_in_edge(1)->get_src();
    auto k_weight = k_matmul->get_in_edge(1)->get_src();
    auto v_weight = v_matmul->get_in_edge(1)->get_src();
    auto qkv_weight_node = MergeQkvWeights(q_weight, k_weight, v_weight, hidden_size, true);

    // 3. merge qkv bias
    auto q_bias = q_add->get_in_edge(1)->get_src();
    auto k_bias = k_add->get_in_edge(1)->get_src();
    auto v_bias = v_add->get_in_edge(1)->get_src();
    auto qkv_bias_node = MergeQkvWeights(q_bias, k_bias, v_bias, hidden_size, false);

    // 4. create broadcast node
    auto input_shape = starting_node->get_output_shape(0);
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
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
    reshape_op->set_name(starting_node->get_name() + "_reshape");
    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {GNodeIndex{starting_node, 0}});

    // 6. create dot node
    auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
    dot_op->set_name(starting_node->get_name() + "_dot");
    auto dot_gnode = m_graph->add_node_and_edge(
        dot_op, {GNodeIndex{reshape_gnode, 0}, GNodeIndex{qkv_weight_node, 0}});

    // 7. create add node
    auto add_op = std::make_shared<op::Add>();
    add_op->set_name(starting_node->get_name() + "_add");
    auto add_gnode = m_graph->add_node_and_edge(
        add_op, {GNodeIndex{dot_gnode, 0}, GNodeIndex{broadcasted_gnode, 0}});

    // 8. create attention node
    nnfusion::op::OpConfig::any myConfig;
    myConfig["num_heads"] = num_heads;
    // myConfig["unidirectional"] = false;
    myConfig["batch_size"] = batch_size;
    myConfig["sequence_length"] = sequence_length;
    // myConfig["past_sequence_length"] = 0;
    myConfig["head_size"] = head_size;
    auto attention_op = std::make_shared<nnfusion::op::GenericOp>(
        starting_node->get_name() + "_attention", "Attention", myConfig);
    auto attention_gnode = m_graph->add_node_and_edge(
        attention_op, {GNodeIndex{add_gnode, 0}, GNodeIndex{mask_index, 0}});

    // add edge
    for (auto out_edge : reshape->get_out_edges())
    {
        if (out_edge->get_dst() == matmul)
        {
            m_graph->add_edge(attention_gnode, 0, out_edge->get_dst(), out_edge->get_dst_input());
            break;
        }
    }

    // remove nodes that are not used anymore
    std::vector<std::shared_ptr<GNode>> nodes_to_remove{
        reshape,  transpose,   qkv_matmul, v_transpose, v_reshape,       v_add,
        v_matmul, qk_div,      qk_matmul,  q_transpose, q_reshape,       q_add,
        q_matmul, k_transpose, k_reshape,  k_add,       k_matmul,        q_weight,
        k_weight, v_weight,    q_bias,     k_bias,      v_bias,          mask_softmax,
        mask_add, mask_mul,    mask_sub,   mask_cast,   mask_unsqueeze2, mask_unsqueeze1};

    for (auto node : nodes_to_remove)
    {
        if (node != nullptr)
            m_graph->remove_node(node);
    }

    return true;
}

bool AttentionFusionOptimizer::ReverseFindPath(
    std::shared_ptr<GNode> node,
    std::vector<std::string>& pattern,
    std::vector<std::vector<std::shared_ptr<GNode>>>& all_paths)
{
    all_paths.clear();
    std::vector<std::shared_ptr<GNode>> path;
    path.push_back(node);
    ReverseSearch(node, pattern, 1, all_paths, path);
    if (all_paths.empty())
        return false;
    for (auto p : all_paths)
    {
        if (p.size() != pattern.size())
            return false;
    }
    return true;
}

void AttentionFusionOptimizer::ReverseSearch(
    std::shared_ptr<GNode> node,
    std::vector<std::string>& pattern,
    size_t idx,
    std::vector<std::vector<std::shared_ptr<GNode>>>& all_paths,
    std::vector<std::shared_ptr<GNode>>& path)
{
    if (idx == pattern.size() && path.size() == pattern.size())
    {
        all_paths.push_back(path);
    }

    for (auto in_edge : node->get_in_edges())
    {
        auto src_node = in_edge->get_src();
        if (src_node->get_op_type() == pattern[idx])
        {
            path.push_back(src_node);
            ReverseSearch(src_node, pattern, idx + 1, all_paths, path);
            path.pop_back();
        }
    }
}

std::shared_ptr<GNode> AttentionFusionOptimizer::MergeQkvWeights(std::shared_ptr<GNode> q_weight,
                                                                 std::shared_ptr<GNode> k_weight,
                                                                 std::shared_ptr<GNode> v_weight,
                                                                 size_t hidden_size,
                                                                 bool is_matmul)
{
    auto q_weight_op = std::dynamic_pointer_cast<op::Constant>(q_weight->get_op_ptr());
    auto k_weight_op = std::dynamic_pointer_cast<op::Constant>(k_weight->get_op_ptr());
    auto v_weight_op = std::dynamic_pointer_cast<op::Constant>(v_weight->get_op_ptr());

    NNFUSION_CHECK_NOT_NULLPTR(q_weight_op);
    NNFUSION_CHECK_NOT_NULLPTR(k_weight_op);
    NNFUSION_CHECK_NOT_NULLPTR(v_weight_op);

    const char* q_weight_dptr = (char*)q_weight_op->get_data_ptr();
    const char* k_weight_dptr = (char*)k_weight_op->get_data_ptr();
    const char* v_weight_dptr = (char*)v_weight_op->get_data_ptr();
    NNFUSION_CHECK_NOT_NULLPTR(q_weight_dptr);
    NNFUSION_CHECK_NOT_NULLPTR(k_weight_dptr);
    NNFUSION_CHECK_NOT_NULLPTR(v_weight_dptr);

    auto weight_dtype = q_weight_op->get_type();

    std::vector<char> qkv_weight_data;
    size_t step = hidden_size * weight_dtype.size();
    if (is_matmul)
        MergeMatMulWeights(q_weight_dptr, k_weight_dptr, v_weight_dptr, step, qkv_weight_data);
    else
        MergeWeights(q_weight_dptr, k_weight_dptr, v_weight_dptr, step, qkv_weight_data);

    nnfusion::Shape qkv_weight_shape{hidden_size, 3 * hidden_size};
    std::shared_ptr<op::Constant> new_qkv_weight_op =
        std::make_shared<op::Constant>(weight_dtype, qkv_weight_shape, qkv_weight_data.data());
    new_qkv_weight_op->set_name("merged_" + q_weight_op->get_name() + "_" +
                                k_weight_op->get_name() + "_" + v_weight_op->get_name());
    std::shared_ptr<GNode> new_qkv_weight_gnode =
        std::make_shared<nnfusion::graph::GNode>(new_qkv_weight_op, GNodeVector());
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
