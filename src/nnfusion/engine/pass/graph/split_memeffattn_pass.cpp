
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "split_memeffattn_pass.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(ftune_output_file);
DEFINE_int32(fblock_size, 128, "");
DEFINE_bool(fsplit_memeffattn, false, "");

bool SplitMemEffAttnPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_ftune_output_file == "" && !FLAGS_fsplit_memeffattn)
        return true;
    NNFUSION_LOG(INFO) << "split MemEffAttn pass starts";
    for (auto node : graph->get_ordered_ops())
    {
        if (node->get_op_type() != "MemEffAttn")
            continue;
        auto op = std::dynamic_pointer_cast<op::GenericOp>(node->get_op_ptr());
        auto q = node->get_in_edge(0)->get_src();
        auto k = node->get_in_edge(1)->get_src();
        auto v = node->get_in_edge(2)->get_src();
        int block_size = FLAGS_fblock_size;
        float softmax_scale = op->localOpConfig.getRoot()["softmax_scale"];
        bool is_causal = op->localOpConfig.getRoot()["is_causal"];
        auto k_shape = node->get_input_shape(1); // b, n, k, d
        size_t K = k_shape[2];
        int num_block = K / block_size;
        Shape q_shape = node->get_input_shape(0);
        size_t B = q_shape[0];
        size_t H = q_shape[1];
        size_t Q = q_shape[2];
        size_t D = q_shape[3];
        nnfusion::element::Type et = node->get_output_element_type(0);
        Shape lse_shape{B, H, Q};
        auto oplse_i = make_shared<op::Variable>(et, lse_shape);
        auto opm_i = make_shared<op::Variable>(et, lse_shape);
        auto opacc_o = make_shared<op::Variable>(et, q_shape);

        auto lse_i = graph->add_node_and_edge(oplse_i, GNodeVector{});
        auto m_i = graph->add_node_and_edge(opm_i, GNodeVector{});
        auto acc_o = graph->add_node_and_edge(opacc_o, GNodeVector{});
        op::OpConfig::any config[6];
        for (int j = 0; j < 6; j++)
        {
            config[j]["stage"] = j;
            config[j]["softmax_scale"] = softmax_scale;
            config[j]["is_causal"] = is_causal;
            config[j]["batch_size"] = B;
            config[j]["num_heads"] = H;
            config[j]["seq_len"] = Q;
            config[j]["head_size"] = D;
            config[j]["seq_len_kv"] = K;
            config[j]["block_size"] = block_size;
        }
        bool single_block = num_block == 1;
        for (int i = 0; i < num_block; i++)
        {
            std::shared_ptr<GNode> kr = k;
            std::shared_ptr<GNode> vr = v;
            if (!single_block)
            {
                Shape lower_bounds({0, 0, 0, 0});
                Shape upper_bounds = k_shape;
                lower_bounds[2] = i * block_size;
                upper_bounds[2] = (i + 1) * block_size;
                auto opkr = make_shared<op::Slice>(lower_bounds, upper_bounds);
                auto opvr = make_shared<op::Slice>(lower_bounds, upper_bounds);
                kr = graph->add_node_and_edge(opkr, {k});
                kr->set_name(k->get_name() + "." + to_string(i));
                vr = graph->add_node_and_edge(opkr, {v});
                vr->set_name(v->get_name() + "." + to_string(i));
            }

            auto opqk = make_shared<op::GenericOp>(
                node->get_name() + "." + to_string(i) + ".qk", "MemEffAttnBasic", config[0]);
            auto opm_i = make_shared<op::GenericOp>(
                node->get_name() + "." + to_string(i) + ".m_i", "MemEffAttnBasic", config[1]);
            auto opp = make_shared<op::GenericOp>(
                node->get_name() + "." + to_string(i) + ".p", "MemEffAttnBasic", config[2]);
            auto opacc_o = make_shared<op::GenericOp>(
                node->get_name() + "." + to_string(i) + ".acc_o", "MemEffAttnBasic", config[3]);
            auto oplse_i = make_shared<op::GenericOp>(
                node->get_name() + "." + to_string(i) + ".lse_i", "MemEffAttnBasic", config[4]);

            auto qk = graph->add_node_and_edge(opqk, {q, kr});
            auto new_m_i = graph->add_node_and_edge(opm_i, {qk, lse_i});
            auto p = graph->add_node_and_edge(opp, {new_m_i, qk});
            auto new_acc_o = graph->add_node_and_edge(opacc_o, {m_i, new_m_i, acc_o, p, vr});
            auto new_lse_i = graph->add_node_and_edge(oplse_i, {new_m_i, lse_i, p});
            m_i = new_m_i;
            lse_i = new_lse_i;
            acc_o = new_acc_o;
        }

        auto opout =
            make_shared<op::GenericOp>(node->get_name() + ".out", "MemEffAttnBasic", config[5]);

        auto out = graph->add_node_and_edge(
            opout, {GNodeIndex(m_i, 0), GNodeIndex(lse_i, 0), GNodeIndex(acc_o, 0)});

        for (auto& edge : node->get_out_edges())
        {
            if (edge->is_control_edge())
                graph->add_control_edge(out, edge->get_dst());
            else
                graph->add_edge(out, 0, edge->get_dst(), edge->get_dst_input());
        }
        NNFUSION_LOG(INFO) << "split MemEffAttn pass ends";
        graph->remove_node(node);
    }
    NNFUSION_LOG(INFO) << "split MemEffAttn pass ends";
    return true;
}
