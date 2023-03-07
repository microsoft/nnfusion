#include "mem_eff_attn_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool MemEffAttnFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        return gnode->get_op_type() == "Softmax";
    };

    SubGraph::Pointer s_attention = std::make_shared<SubGraph>();
    s_attention->name = "attention";
    s_attention->check_starting_node = check_root;

    //path_q #1
    {
        Pattern::Pointer p_q = std::make_shared<Pattern>();
        std::vector<std::string> ops_q1{
            "Softmax",
            "Add",
            "Multiply",
            "Reshape",
            "BatchMatMul",
            "Broadcast",
            "Reshape",
            "Reshape",
            "Reshape",
        };

        std::vector<std::string> ops_q2{
            "Softmax",
            "Multiply",
            "Reshape",
            "BatchMatMul",
            "Broadcast",
            "Reshape",
            "Reshape",
            "Reshape",
        };

        p_q->descriptions.push_back(std::make_pair(ops_q1, 4));
        p_q->descriptions.push_back(std::make_pair(ops_q2, 3));
        p_q->reverse_order = true;

        auto check_q = [](const PatternRecord& pr) -> bool {
            size_t path_q_len = pr.nodes.size();
            auto mul = pr.nodes[path_q_len - 7];
            std::shared_ptr<GNode> broadcast;
            for (size_t i = 0; i < 2; i++)
            {
                auto edge = mul->get_in_edge(i);
                auto src = edge->get_src();
                if (src->get_op_type() == "Broadcast")
                {
                    broadcast = src;
                    break;
                }
            }

            if (!broadcast)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q";
                return false;
            }

            if (!broadcast->get_in_edge(0)->get_src()->is_constant())
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path q";
                return false;
            }

            auto q = pr.nodes.back();
            if (q->get_output_shape(0).size() != 4)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << q->get_name() << q->get_output_shape(0)
                                               << "Failed to find path q";
                return false;
            }
            return true;
        };
        p_q->check.push_back(check_q);
        s_attention->patterns.push_back(p_q);
    }

    //path_k
    {
        Pattern::Pointer p_k = std::make_shared<Pattern>();
        std::vector<std::string> ops_k{
            "BatchMatMul", "Broadcast", "Reshape", "Reshape", "Reshape", "Reshape"};

        p_k->descriptions.push_back(std::make_pair(ops_k, 0));
        p_k->reverse_order = true;

        auto check_k = [](const PatternRecord& pr) -> bool {
            auto k = pr.nodes.back();
            if (k->get_output_shape(0).size() != 4)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path k";
                return false;
            }
            return true;
        };
        p_k->check.push_back(check_k);
        s_attention->patterns.push_back(p_k);
    }

    //path_end
    {
        Pattern::Pointer p_end = std::make_shared<Pattern>();
        std::vector<std::string> ops_end1{"BatchMatMul",
                                          "Reshape",
                                          "Multiply",
                                          "Add",
                                          "Softmax",
                                          "Reshape",
                                          "Broadcast",
                                          "BatchMatMul",
                                          "Reshape",
                                          "Reshape",
                                          "Reshape"};
        std::vector<std::string> ops_end2{"BatchMatMul",
                                          "Reshape",
                                          "Multiply",
                                          "Softmax",
                                          "Reshape",
                                          "Broadcast",
                                          "BatchMatMul",
                                          "Reshape",
                                          "Reshape",
                                          "Reshape"};

        p_end->descriptions.push_back(std::make_pair(ops_end1, 7));
        p_end->descriptions.push_back(std::make_pair(ops_end2, 6));
        p_end->reverse_order = false;

        auto check_end = [](const PatternRecord& pr) -> bool {
            auto end = pr.nodes.back();
            if (end->get_output_shape(0).size() != 4)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path end";
                return false;
            }
            return true;
        };
        p_end->check.push_back(check_end);
        s_attention->patterns.push_back(p_end);
    }

    //path_v
    {
        Pattern::Pointer p_v = std::make_shared<Pattern>();
        std::vector<std::string> ops_v{"BatchMatMul", "Broadcast", "Reshape", "Reshape", "Reshape"};

        p_v->descriptions.push_back(std::make_pair(ops_v, 1));
        p_v->reverse_order = true;

        auto check_v = [](const PatternRecord& pr) -> bool {
            auto v = pr.nodes.back();
            if (v->get_output_shape(0).size() != 4)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path v";
                return false;
            }
            return true;
        };
        p_v->check.push_back(check_v);

        s_attention->patterns.push_back(p_v);
    }

    m_subgraphs.push_back(s_attention);
    return true;
}

bool MemEffAttnFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto starting_node = subgraph_record->get_starting_node();
    auto pr_q = subgraph_record->pattern_records[0];
    auto pr_k = subgraph_record->pattern_records[1];
    auto pr_end = subgraph_record->pattern_records[2];
    auto pr_v = subgraph_record->pattern_records[3];

    auto q = pr_q->nodes.back();
    auto k = pr_k->nodes.back();
    auto v = pr_v->nodes.back();
    auto out = pr_end->nodes.back();

    auto mul = pr_end->nodes[2];
    std::shared_ptr<GNode> broadcast, scale;
    // for (size_t i = 0; i < 2; i++)
    // {
    //     auto edge = mul->get_in_edge(i);
    //     auto src = edge->get_src();
    //     if (src->get_op_type() == "Broadcast")
    //     {
    //         broadcast = src;
    //         break;
    //     }
    // }

    // if (!broadcast)
    // {
    //     NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find scale";
    //     return false;
    // }
    // auto in_edge = broadcast->get_in_edge(0);

    // scale = in_edge->get_src();

    // if (!scale->is_constant())
    // {
    //     NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find scale";
    //     return false;
    // }

    nnfusion::op::OpConfig::any myConfig;
    // if (scale->get_element_type() == element::f16)
    // {
    //     std::vector<element::half> scale_value;
    //     bool status = nnfusion::frontend::GetValueFromNGraphOp<element::half>(scale, &scale_value);
    //     if (!status || scale_value.size() != 1)
    //     {
    //         NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find scale";
    //         return false;
    //     }
    //     myConfig["softmax_scale"] = scale_value[0];
    // }
    // else
    // {
    //     std::vector<float> scale_value;
    //     bool status = nnfusion::frontend::GetValueFromNGraphOp<float>(scale, &scale_value);
    //     if (!status || scale_value.size() != 1)
    //     {
    //         NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
    //         return false;
    //     }
    //     myConfig["softmax_scale"] = scale_value[0];
    // }

    auto qshape = q->get_output_shape(0);
    auto vshape = v->get_output_shape(0);
    myConfig["batch_size"] = qshape[0];
    myConfig["seq_len"] = qshape[2];
    myConfig["seq_len_kv"] = vshape[2];
    myConfig["num_heads"] = qshape[1];
    myConfig["head_size"] = qshape[3];
    myConfig["head_size_v"] = vshape[3];

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;

    auto attention_op = std::make_shared<nnfusion::op::GenericOp>(
        starting_node->get_name() + "_memeffattn", "MemEffAttn", myConfig);
    auto attention_gnode = graph->add_node_and_edge(
        attention_op, {GNodeIndex{q, 0}, GNodeIndex{k, 0}, GNodeIndex{v, 0}});

    std::shared_ptr<GNode> last_node = out;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(attention_gnode, 0, dst, y);
    }

    nodes_to_remove.insert(pr_v->nodes.begin(), pr_v->nodes.end() - 1);
    nodes_to_remove.insert(pr_q->nodes.begin(), pr_q->nodes.end() - 1);
    nodes_to_remove.insert(pr_k->nodes.begin(), pr_k->nodes.end() - 1);
    nodes_to_remove.insert(pr_end->nodes.begin(), pr_end->nodes.end());
    nodes_to_remove.insert({scale, broadcast});

    std::shared_ptr<GNode> add = pr_end->nodes[3];
    if (add->get_op_type() == "Add")
    {
        nodes_to_remove.insert(add);
        std::shared_ptr<GNode> add_const;
        for (size_t i = 0; i < 2; i++)
        {
            auto edge = add->get_in_edge(i);
            auto src = edge->get_src();
            if (src->is_constant())
            {
                add_const = src;
                break;
            }
        }
        if (add_const)
            nodes_to_remove.insert(add_const);
    }
    NNFUSION_LOG(INFO) << attention_gnode->get_name();
    if (!RemoveNodes(nodes_to_remove, attention_gnode))
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "remove nodes failed.";
        return false;
    }

    return true;
}
