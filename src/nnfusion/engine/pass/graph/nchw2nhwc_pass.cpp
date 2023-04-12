
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nchw2nhwc_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_int64(fnchw2nhwc_level);

std::shared_ptr<GNode> NCHW2NHWCPass::add_transpose(std::shared_ptr<Graph>& graph,
                                                    std::shared_ptr<GNode> node,
                                                    bool to_nhwc)
{
    NNFUSION_LOG(INFO) << node->get_name() << to_nhwc;
    auto input_shape = node->get_output_shape(0);
    std::string convert_type;
    nnfusion::AxisVector axis_order;
    nnfusion::Shape reshape_outshape;
    if (to_nhwc)
    {
        convert_type = "_2nhwc";
        axis_order = {0, 2, 3, 1};
        reshape_outshape = {input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
    }
    else
    {
        convert_type = "_2nchw";
        axis_order = {0, 3, 1, 2};
        reshape_outshape = {input_shape[0], input_shape[3], input_shape[1], input_shape[2]};
    }

    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(axis_order, reshape_outshape);
    auto reshape_gnode = graph->add_node_and_edge(reshape_op, {GNodeIndex{node, 0}});
    reshape_gnode->set_name(node->get_name() + convert_type);
    return reshape_gnode;
}

void NCHW2NHWCPass::remove_node(std::shared_ptr<Graph>& graph,
                                std::shared_ptr<nnfusion::graph::GNode> node)
{
    nnfusion::graph::GNodeVector updated_outputs;
    for (auto out : graph->get_outputs())
    {
        if (out != node)
        {
            updated_outputs.push_back(out);
        }
        else
        {
            updated_outputs.push_back(node);
        }
    }
    graph->set_outputs(updated_outputs);
    graph->remove_node(node);
    return;
}

bool NCHW2NHWCPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_fnchw2nhwc_level == 0)
        return true;
    else
    {
        NNFUSION_LOG(INFO) << "NCHW2NHWC pass Start";
        for (auto node : graph->get_ordered_ops())
        {
            if (node->get_op_type() == "Convolution" && node->get_output_shape(0).size() == 4)
            {
                std::shared_ptr<op::Convolution> op =
                    std::dynamic_pointer_cast<op::Convolution>(node->get_op_ptr());
                if (op->get_data_format() != "NCHW")
                    continue;
                auto data = node->get_in_edge(0)->get_src();
                auto filter = node->get_in_edge(1)->get_src();
                const auto& strides = op->get_window_movement_strides();
                const auto& pads = op->get_padding_below();
                const auto& dilations = op->get_window_dilation_strides();

                op::OpConfig::any config;
                config["strides"] = strides;
                config["dilations"] = dilations;
                config["pads"] = pads;
                auto nhwcconv_op = std::make_shared<op::GenericOp>(
                    node->get_name() + "_nhwcconv", "NhwcConv", config);
                auto data_nhwc = this->add_transpose(graph, data, true);
                auto filter_nhwc = this->add_transpose(graph, filter, true);
                std::shared_ptr<nnfusion::graph::GNode> nhwcconv_node = graph->add_node_and_edge(
                    nhwcconv_op, {GNodeIndex{data_nhwc, 0}, GNodeIndex{filter_nhwc, 0}});
                auto conv_2nchw = this->add_transpose(graph, nhwcconv_node, false);
                for (auto& edge : node->get_out_edges())
                {
                    if (edge->is_control_edge())
                        graph->add_control_edge(conv_2nchw, edge->get_dst());
                    else
                        graph->add_edge(conv_2nchw, 0, edge->get_dst(), edge->get_dst_input());
                }
                this->remove_node(graph, node);
            }
        }
        NNFUSION_LOG(INFO) << "NCHW2NHWC pass ends";
        return true;
    }
}