
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "tensor_core_rewrite_pass.hpp"
#include "nnfusion/core/operators/op_define/convolution.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(ftune_output_file);

bool TensorCoreRewritePass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_ftune_output_file == "")
        return true;
    for (auto node : graph->get_ordered_ops()) {
        if (node->get_op_type() != "Convolution") continue;
        if (node->get_element_type() != element::f16) continue;
        auto op = std::dynamic_pointer_cast<op::Convolution>(node->get_op_ptr());
        auto out_shape = node->get_output_shape(0);
        size_t N = out_shape[0], C = out_shape[1], H = out_shape[2], W = out_shape[2];
        size_t in_channel = node->get_input_shape(0)[1];
        if (C % 8 > 0 || N*H*W % 8 > 0 || N*C*H*W % 256 > 0 || in_channel % 16 > 0) continue;
        op::OpConfig::any config, config2;
        config["N"] = N, config["C"] = C, config["H"] = H, config["W"] = W;
        const auto& stride_h = op->get_window_movement_strides()[0];
        const auto& stride_w = op->get_window_movement_strides()[1];
        const auto& is_nchw = op->get_data_format() == "NCHW";
        const auto& padding_below = op->get_padding_below();
        const auto& padding_above = op->get_padding_above();
        const auto& padding_h = op->get_padding_below()[0];
        const auto& padding_w = op->get_padding_below()[1];
        const auto& dilation_h = op->get_window_dilation_strides()[0];
        const auto& dilation_w = op->get_window_dilation_strides()[1];
        NNFUSION_CHECK(is_nchw);
        NNFUSION_CHECK(padding_h == padding_w);
        NNFUSION_CHECK(stride_h == stride_w);
        NNFUSION_CHECK(dilation_h == dilation_w);
        config["S"] = stride_h;
        config["D"] = dilation_h;
        config["P"] = padding_h;
        auto op0 = make_shared<op::GenericOp>(node->get_name()+".tc", "ImplicitGemm", config);
        auto op1 = make_shared<op::GenericOp>(node->get_name()+".reshape", "CNHW2NCHW", config);
        auto data = node->get_in_edge(0)->get_src();
        auto weight = node->get_in_edge(1)->get_src();
        auto node0 = graph->add_node_and_edge(op0, {data, weight});
        auto node1 = graph->add_node_and_edge(op1, {node0});
        for (auto& edge : node->get_out_edges())
        {
            if (edge->is_control_edge())
                graph->add_control_edge(node1, edge->get_dst());
            else
                graph->add_edge(node1, 0, edge->get_dst(), edge->get_dst_input());
        }
        graph->remove_node(node);
    }
    return true;
}
