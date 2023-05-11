
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
    NNFUSION_LOG(INFO) << "split Attn pass starts";
    for (auto node : graph->get_ordered_ops())
    {
        if (node->get_op_type() == "MemEffAttn")
        {
            auto op = std::dynamic_pointer_cast<op::GenericOp>(node->get_op_ptr());
            auto q = node->get_in_edge(0)->get_src();
            auto k = node->get_in_edge(1)->get_src();
            auto v = node->get_in_edge(2)->get_src();
            auto lse = node->get_in_edge(3)->get_src();
            auto m = node->get_in_edge(4)->get_src();
            auto acco = node->get_in_edge(5)->get_src();

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
            op::OpConfig::any identity_config[3];
            Shape lse_shape{B, H, Q};
            auto oplse_i = make_shared<op::GenericOp>("lse0", "Identity", identity_config[0]);
            auto opm_i = make_shared<op::GenericOp>("m0", "Identity", identity_config[1]);
            auto opacc_o = make_shared<op::GenericOp>("acc_o0", "Identity", identity_config[2]);

            auto lse_i = graph->add_node_and_edge(oplse_i, {lse});
            auto m_i = graph->add_node_and_edge(opm_i, {m});
            auto acc_o = graph->add_node_and_edge(opacc_o, {acco});
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
            NNFUSION_LOG(INFO) << "split MemEffAttn";
            graph->remove_node(node);
        }

        else if (node->get_op_type() == "MultiScaleAttn0")
        {
            auto op = std::dynamic_pointer_cast<op::GenericOp>(node->get_op_ptr());
            auto qr = node->get_in_edge(0)->get_src();
            auto kr = node->get_in_edge(1)->get_src();
            auto v = node->get_in_edge(2)->get_src();
            auto mask = node->get_in_edge(3)->get_src();

            Shape qr_shape = node->get_input_shape(0);
            Shape v_shape = node->get_input_shape(2);
            // size_t B = qr_shape[0];
            size_t BNBL = qr_shape[0];
            size_t NQ = qr_shape[1];
            size_t BLQ = qr_shape[2];
            size_t KD = qr_shape[3];
            size_t NV = v_shape[2];
            size_t BLK = v_shape[3];
            size_t D = v_shape[4];
            nnfusion::element::Type et = node->get_output_element_type(0);

            op::OpConfig::any config[3];
            for (int j = 0; j < 3; j++)
            {
                config[j]["stage"] = j;
                // config[j]["b"] = B;
                config[j]["bnbl"] = BNBL;
                config[j]["nq"] = NQ;
                config[j]["blq"] = BLQ;
                config[j]["kd"] = KD;
                config[j]["nv"] = NV;
                config[j]["blk"] = BLK;
                config[j]["d"] = D;
            }
            auto opqk = make_shared<op::GenericOp>(
                node->get_name() + ".qk", "MultiScaleAttnBasic0", config[0]);
            auto opqkm = make_shared<op::GenericOp>(
                node->get_name() + ".qkm", "MultiScaleAttnBasic0", config[1]);
            auto opattn = make_shared<op::GenericOp>(
                node->get_name() + ".attn", "MultiScaleAttnBasic0", config[2]);

            auto qk = graph->add_node_and_edge(opqk, {qr, kr});
            auto qkm = graph->add_node_and_edge(opqkm, {qk, mask});
            auto out = graph->add_node_and_edge(opattn, {qkm, v});

            for (auto& edge : node->get_out_edges())
            {
                if (edge->is_control_edge())
                    graph->add_control_edge(out, edge->get_dst());
                else
                    graph->add_edge(out, 0, edge->get_dst(), edge->get_dst_input());
            }
            NNFUSION_LOG(INFO) << "split MultiScaleAttn0";
            graph->remove_node(node);
        }

        else if (node->get_op_type() == "MultiScaleAttn1")
        {
            auto op = std::dynamic_pointer_cast<op::GenericOp>(node->get_op_ptr());
            auto qr = node->get_in_edge(0)->get_src();
            auto kr = node->get_in_edge(1)->get_src();
            auto v = node->get_in_edge(2)->get_src();
            auto mask = node->get_in_edge(3)->get_src();
            auto cross_decay = node->get_in_edge(4)->get_src();
            auto inner_decay = node->get_in_edge(5)->get_src();
            auto kv_state = node->get_in_edge(6)->get_src();

            Shape qr_shape = node->get_input_shape(0);
            Shape v_shape = node->get_input_shape(2);
            size_t B = qr_shape[0];
            size_t NQ = qr_shape[1];
            size_t BLQ = qr_shape[2];
            size_t KD = qr_shape[3];
            size_t NV = v_shape[2];
            size_t BLK = v_shape[3];
            size_t D = v_shape[4];

            op::OpConfig::any config[4];
            for (int j = 0; j < 4; j++)
            {
                config[j]["stage"] = j;
                config[j]["b"] = B;
                config[j]["nq"] = NQ;
                config[j]["blq"] = BLQ;
                config[j]["kd"] = KD;
                config[j]["nv"] = NV;
                config[j]["blk"] = BLK;
                config[j]["d"] = D;
            }
            auto opvrm = make_shared<op::GenericOp>(
                node->get_name() + ".vrm", "MultiScaleAttnBasic1", config[0]);
            auto opkv = make_shared<op::GenericOp>(
                node->get_name() + ".kv", "MultiScaleAttnBasic1", config[1]);
            auto opnew_kv_state = make_shared<op::GenericOp>(
                node->get_name() + ".new_kv_state", "MultiScaleAttnBasic1", config[2]);
            auto opout = make_shared<op::GenericOp>(
                node->get_name() + ".crossattn", "MultiScaleAttnBasic1", config[3]);

            auto vrm = graph->add_node_and_edge(opvrm, {v, mask});
            auto kv = graph->add_node_and_edge(opkv, {kr, vrm});
            auto new_kv_state =
                graph->add_node_and_edge(opnew_kv_state, {kv_state, cross_decay, kv});
            auto out = graph->add_node_and_edge(opout, {qr, new_kv_state, inner_decay});

            for (auto& edge : node->get_out_edges())
            {
                if (edge->is_control_edge())
                    graph->add_control_edge(out, edge->get_dst());
                else
                    graph->add_edge(out, 0, edge->get_dst(), edge->get_dst_input());
            }
            NNFUSION_LOG(INFO) << "split MultiScaleAttn1";
            graph->remove_node(node);
        }

        else if (node->get_op_type() == "MultiScaleAttn")
        {
            auto op = std::dynamic_pointer_cast<op::GenericOp>(node->get_op_ptr());
            auto qr = node->get_in_edge(0)->get_src();
            auto kr = node->get_in_edge(1)->get_src();
            auto v = node->get_in_edge(2)->get_src();
            auto mask = node->get_in_edge(3)->get_src();
            auto cross_decay = node->get_in_edge(4)->get_src();
            auto inner_decay = node->get_in_edge(5)->get_src();
            auto kv_state = node->get_in_edge(6)->get_src();

            Shape qr_shape = node->get_input_shape(0);
            Shape v_shape = node->get_input_shape(2);
            size_t B = qr_shape[0];
            size_t NQ = qr_shape[1];
            size_t BLQ = qr_shape[2];
            size_t KD = qr_shape[3];
            size_t NV = v_shape[2];
            size_t BLK = v_shape[3];
            size_t D = v_shape[4];

            op::OpConfig::any config[8];
            for (int j = 0; j < 8; j++)
            {
                config[j]["stage"] = j;
                config[j]["b"] = B;
                config[j]["nq"] = NQ;
                config[j]["blq"] = BLQ;
                config[j]["kd"] = KD;
                config[j]["nv"] = NV;
                config[j]["blk"] = BLK;
                config[j]["d"] = D;
            }
            auto opqk = make_shared<op::GenericOp>(
                node->get_name() + ".qk", "MultiScaleAttnBasic", config[0]);
            auto opqkm = make_shared<op::GenericOp>(
                node->get_name() + ".qkm", "MultiScaleAttnBasic", config[1]);
            auto opattn = make_shared<op::GenericOp>(
                node->get_name() + ".attn", "MultiScaleAttnBasic", config[2]);
            auto opvrm = make_shared<op::GenericOp>(
                node->get_name() + ".vrm", "MultiScaleAttnBasic", config[3]);
            auto opkv = make_shared<op::GenericOp>(
                node->get_name() + ".kv", "MultiScaleAttnBasic", config[4]);
            auto opnew_kv_state = make_shared<op::GenericOp>(
                node->get_name() + ".new_kv_state", "MultiScaleAttnBasic", config[5]);
            auto opcrossattn = make_shared<op::GenericOp>(
                node->get_name() + ".crossattn", "MultiScaleAttnBasic", config[6]);
            auto opout = make_shared<op::GenericOp>(
                node->get_name() + ".out", "MultiScaleAttnBasic", config[7]);

            auto qk = graph->add_node_and_edge(opqk, {qr, kr});
            auto qkm = graph->add_node_and_edge(opqkm, {qk, mask});
            auto attn = graph->add_node_and_edge(opattn, {qkm, v});
            auto vrm = graph->add_node_and_edge(opvrm, {v, mask});
            auto kv = graph->add_node_and_edge(opkv, {kr, vrm});
            auto new_kv_state =
                graph->add_node_and_edge(opnew_kv_state, {kv_state, cross_decay, kv});
            auto crossattn = graph->add_node_and_edge(opcrossattn, {qr, new_kv_state, inner_decay});
            auto out = graph->add_node_and_edge(opout, {attn, crossattn});

            for (auto& edge : node->get_out_edges())
            {
                if (edge->is_control_edge())
                    graph->add_control_edge(out, edge->get_dst());
                else
                    graph->add_edge(out, 0, edge->get_dst(), edge->get_dst_input());
            }
            NNFUSION_LOG(INFO) << "split MultiScaleAttn0";
            graph->remove_node(node);
        }
    }
    NNFUSION_LOG(INFO) << "split Attn pass ends";
    return true;
}
