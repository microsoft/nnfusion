#include "attention_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool AttentionFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        // Starting Node has 4 children
        if (gnode->get_out_edges().size() != 4)
            return false;
        // Starting Node shape = [batch_size, sequence_length, hidden_size]
        auto shape = gnode->get_output_shape(0);
        if (shape.size() != 3)
            return false;
        auto hidden_size = shape[2];

        // Starting Node has 4 children:  3 of them are Dot
        auto edges = gnode->get_out_edges();
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
    };

    SubGraph::Pointer s_attention = std::make_shared<SubGraph>();
    s_attention->name = "attention";
    s_attention->check_starting_node = check_root;

    //path_end #0
    {
        Pattern::Pointer p_end = std::make_shared<Pattern>();
        std::vector<std::string> ops_end{"AnyOp", "AnyOp"};
        p_end->descriptions.push_back(std::make_pair(ops_end, 1));
        p_end->reverse_order = false;
        auto check_end = [](const PatternRecord& pr) -> bool {
            // check the end node is not Dot
            auto end = pr.nodes[1];
            return (end->get_op_type() != "Dot");
        };
        p_end->check.push_back(check_end);

        s_attention->patterns.push_back(p_end);
    }

    //path_v #1
    {
        Pattern::Pointer p_v = std::make_shared<Pattern>();
        std::vector<std::string> ops_v1{"AnyOp",
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
                                        "Dot"};

        std::vector<std::string> ops_v2{"AnyOp",
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
                                        "Dot"};
        p_v->descriptions.push_back(std::make_pair(ops_v1, 6));
        p_v->descriptions.push_back(std::make_pair(ops_v2, 5));
        p_v->reverse_order = true;
        auto check_v = [](const PatternRecord& pr) -> bool {
            // check the end node is not Dot
            size_t path_v_len = pr.nodes.size();
            auto reshape2 = pr.nodes[path_v_len - 10];
            auto qkv_batmatmul = pr.nodes[path_v_len - 7];
            auto v_reshape2 = pr.nodes[path_v_len - 4];
            auto v_reshape1 = pr.nodes[path_v_len - 3];
            auto v_add = pr.nodes[path_v_len - 2];
            auto v_dot = pr.nodes[path_v_len - 1];
            auto starting_node = pr.nodes.back();

            auto v_reshape1_shape = v_reshape1->get_output_shape(0);

            if (v_add->get_out_edges().size() != 1 || v_dot->get_out_edges().size() != 1 ||
                v_reshape2->get_out_edges().size() != 1 ||
                qkv_batmatmul->get_out_edges().size() != 1 ||
                v_reshape2->get_out_edges().size() != 1 ||
                v_reshape1->get_out_edges().size() != 1 || v_reshape1_shape.size() != 4)
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

            return true;
        };
        p_v->check.push_back(check_v);

        s_attention->patterns.push_back(p_v);
    }

    //path_mask #2
    {
        Pattern::Pointer p_mask = std::make_shared<Pattern>();
        std::vector<std::string> ops_mask1{"BatchMatMul",
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
        std::vector<std::string> ops_mask2{"BatchMatMul",
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
        p_mask->descriptions.push_back(std::make_pair(ops_mask1, 4));
        p_mask->descriptions.push_back(std::make_pair(ops_mask2, 4));
        p_mask->reverse_order = true;

        s_attention->patterns.push_back(p_mask);
    }

    //path_q #3
    {
        Pattern::Pointer p_q = std::make_shared<Pattern>();
        std::vector<std::string> ops_q{"Add",
                                       "Divide",
                                       "Reshape",
                                       "BatchMatMul",
                                       "Broadcast",
                                       "Reshape",
                                       "Reshape",
                                       "Reshape",
                                       "Add",
                                       "Dot"};

        p_q->descriptions.push_back(std::make_pair(ops_q, 3));
        p_q->reverse_order = true;
        auto check_q = [](const PatternRecord& pr) -> bool {
            auto qk_batmatmul = pr.nodes[3];
            auto q_broadcast = pr.nodes[4];
            if (q_broadcast != qk_batmatmul->get_in_edge(0)->get_src())
                return false;

            return true;
        };
        p_q->check.push_back(check_q);

        s_attention->patterns.push_back(p_q);
    }

    //path_k #4
    {
        Pattern::Pointer p_k = std::make_shared<Pattern>();
        std::vector<std::string> ops_k{
            "BatchMatMul", "Broadcast", "Reshape", "Reshape", "Reshape", "Add", "Dot"};

        p_k->descriptions.push_back(std::make_pair(ops_k, 1));
        p_k->reverse_order = true;
        auto check_k = [](const PatternRecord& pr) -> bool {
            auto qk_batmatmul = pr.nodes[0];
            auto k_broadcast = pr.nodes[1];
            if (k_broadcast != qk_batmatmul->get_in_edge(1)->get_src())
                return false;

            return true;
        };
        p_k->check.push_back(check_k);

        s_attention->patterns.push_back(p_k);
    }

    auto check_attention1 = [](const SubGraphRecord& sr) -> bool {
        auto starting_node = sr.get_starting_node();
        //v
        auto pr_v = sr.pattern_records[1];
        size_t path_v_len = pr_v->nodes.size();
        auto v_reshape1 = pr_v->nodes[path_v_len - 3];
        auto v_reshape1_shape = v_reshape1->get_output_shape(0);
        size_t num_heads = v_reshape1_shape[2];
        size_t head_size = v_reshape1_shape[3];
        size_t hidden_size = starting_node->get_output_shape(0)[2];
        if (hidden_size != num_heads * head_size)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path v";
            return false;
        }

        auto v_dot = pr_v->nodes.back();
        if (v_dot->get_in_edge(0)->get_src() != starting_node)
            return false;
        // q
        auto pr_q = sr.pattern_records[3];
        auto q_dot = pr_q->nodes.back();
        if (q_dot->get_in_edge(0)->get_src() != starting_node)
            return false;
        auto q_reshape1 = pr_q->nodes[7];
        auto q_reshape1_shape = q_reshape1->get_output_shape(0);
        if (q_reshape1_shape.size() != 4 || q_reshape1_shape[2] != num_heads ||
            q_reshape1_shape[3] != head_size)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q";
            return false;
        }
        //k
        auto pr_k = sr.pattern_records[4];
        auto k_dot = pr_k->nodes.back();
        if (k_dot->get_in_edge(0)->get_src() != starting_node)
            return false;
        return true;
    };
    s_attention->check.push_back(check_attention1);

    m_subgraphs.push_back(s_attention);
    return true;
}

bool AttentionFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto starting_node = subgraph_record->get_starting_node();
    auto pr_end = subgraph_record->pattern_records[0];
    auto pr_v = subgraph_record->pattern_records[1];
    auto pr_mask = subgraph_record->pattern_records[2];
    auto pr_q = subgraph_record->pattern_records[3];
    auto pr_k = subgraph_record->pattern_records[4];
    auto mask_reshape1 = pr_mask->nodes.back();
    auto q_dot = pr_q->nodes.back();
    auto k_dot = pr_k->nodes.back();
    auto v_dot = pr_v->nodes.back();
    auto q_add = pr_q->nodes[pr_q->nodes.size() - 2];
    auto k_add = pr_k->nodes[pr_k->nodes.size() - 2];
    auto v_add = pr_v->nodes[pr_v->nodes.size() - 2];
    auto v_reshape1 = pr_v->nodes[pr_v->nodes.size() - 3];
    auto v_reshape1_shape = v_reshape1->get_output_shape(0);
    size_t num_heads = v_reshape1_shape[2];
    size_t head_size = v_reshape1_shape[3];
    auto input_shape = starting_node->get_output_shape(0);
    size_t batch_size = input_shape[0];
    size_t sequence_length = input_shape[1];
    auto reshape2 = pr_v->nodes[pr_v->nodes.size() - 10];
    auto mask_broadcast_for_add = pr_mask->nodes[5];

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;

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
    size_t hidden_size = starting_node->get_output_shape(0)[2];
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
            nodes_to_remove.insert(weight);
        }
    }

    // 3. merge qkv bias
    auto q_bias_broadcast = q_add->get_in_edge(1)->get_src();
    auto q_bias = q_bias_broadcast->get_in_edge(0)->get_src();
    auto k_bias_broadcast = k_add->get_in_edge(1)->get_src();
    auto k_bias = k_bias_broadcast->get_in_edge(0)->get_src();
    auto v_bias_broadcast = v_add->get_in_edge(1)->get_src();
    auto v_bias = v_bias_broadcast->get_in_edge(0)->get_src();
    //todo: remove below after standalize graph
    if (q_bias->get_op_type() == "Reshape")
    {
        q_bias = q_bias->get_in_edge(0)->get_src();
    }
    if (k_bias->get_op_type() == "Reshape")
    {
        k_bias = k_bias->get_in_edge(0)->get_src();
    }
    if (v_bias->get_op_type() == "Reshape")
    {
        v_bias = v_bias->get_in_edge(0)->get_src();
    }
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
            nodes_to_remove.insert(bias);
        }
    }

    // 4. create broadcast node
    auto broadcasted_op = std::make_shared<op::Broadcast>(
        nnfusion::Shape({batch_size * sequence_length, 3 * hidden_size}), nnfusion::AxisSet({0}));
    broadcasted_op->set_name(qkv_bias_node->get_name() + "_broadcast");
    auto broadcasted_gnode =
        graph->add_node_and_edge(broadcasted_op, {GNodeIndex{qkv_bias_node, 0}});

    // 5. create reshape node
    nnfusion::AxisVector ng_axis_order(input_shape.size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(
        ng_axis_order, nnfusion::Shape({batch_size * sequence_length, hidden_size}));
    reshape_op->set_name(starting_node->get_name() + "_reshape");
    auto reshape_gnode = graph->add_node_and_edge(reshape_op, {GNodeIndex{starting_node, 0}});

    // 6. create dot node
    auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
    dot_op->set_name(starting_node->get_name() + "_dot");
    auto dot_gnode = graph->add_node_and_edge(
        dot_op, {GNodeIndex{reshape_gnode, 0}, GNodeIndex{qkv_weight_node, 0}});

    // 7. create add node
    auto add_op = std::make_shared<op::Add>();
    add_op->set_name(starting_node->get_name() + "_add");
    auto add_gnode = graph->add_node_and_edge(
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
        starting_node->get_name() + "_attention", "Attention", myConfig);
    auto attention_gnode = graph->add_node_and_edge(
        attention_op, {GNodeIndex{add_gnode, 0}, GNodeIndex{mask_index, 0}});

    std::shared_ptr<GNode> last_node = reshape2;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(attention_gnode, 0, dst, y);
    }

    nodes_to_remove.insert(pr_v->nodes.end() - 10, pr_v->nodes.end());
    nodes_to_remove.insert(pr_mask->nodes.begin() + 1, pr_mask->nodes.begin() + 5);
    if (mask_broadcast_for_add->get_out_edges().size() == 1)
        nodes_to_remove.insert(pr_mask->nodes.begin() + 5, pr_mask->nodes.end());
    nodes_to_remove.insert(pr_q->nodes.begin() + 1, pr_q->nodes.end());
    nodes_to_remove.insert(pr_k->nodes.begin(), pr_k->nodes.end());

    if (!RemoveNodes(nodes_to_remove, attention_gnode))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "remove nodes failed.";
        return false;
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
        std::shared_ptr<GNode> cast_gnode = graph->add_node_and_edge(cast_op, {mask_input});
        mask_index = cast_gnode;
    }

    mask_index_map.insert(
        std::pair<std::string, std::shared_ptr<GNode>>(mask_input->get_name(), mask_index));
    return mask_index;
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
    // NNFUSION_LOG(INFO) << q_weight->get_name() << ", " << k_weight->get_name() << ", "
    //                    << v_weight->get_name();
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
        graph->add_node_and_edge(new_qkv_weight_op, GNodeVector());
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