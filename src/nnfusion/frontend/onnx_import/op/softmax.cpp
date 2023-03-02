//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "softmax.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "util/reshape.hpp"

DEFINE_bool(fsplit_softmax, false, "decompose softmax into multiple ops");

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSoftmaxOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    Node node(node_proto);
                    auto axis = node.get_attribute_value<std::vector<int64_t>>("axis", {-1})[0];
                    axis += axis < 0 ? input_index.get_shape().size() : 0;
                    nnfusion::AxisSet ng_axes_softmax;
                    ng_axes_softmax.insert(axis);
                    if (!FLAGS_fsplit_softmax)
                    {
                        auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                        softmax_op->set_name(node_proto.output(0));
                        auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {input_index});
                        NamedNodeVector ret{{node_proto.output(0), softmax_gnode}};
                        return ret;
                    }
                    else
                    {
                        // Max
                        auto max_op = std::make_shared<op::Max>(ng_axes_softmax);
                        max_op->set_name(node_proto.output(0) + "_max");
                        auto max_gnode = m_graph->add_node_and_edge(max_op, {input_index});

                        // Reshape
                        auto max_shape = max_gnode->get_output_shape(0);
                        auto extend_max_shape = max_shape;
                        for (auto axis : ng_axes_softmax)
                        {
                            extend_max_shape.insert(std::next(extend_max_shape.begin(), axis), 1);
                        }
                        nnfusion::AxisVector ng_axis_order(max_shape.size());
                        std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                        auto reshape_op =
                            std::make_shared<op::Reshape>(ng_axis_order, extend_max_shape);
                        reshape_op->set_name(node_proto.output(0) + "_reshape");
                        auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {max_gnode});

                        //Broadcast
                        auto max_index = GNodeIndex(reshape_gnode);
                        std::tie(input_index, max_index) =
                            graph::numpy_broadcast(std::make_pair(input_index, max_index), m_graph);

                        // Sub
                        auto sub_op = std::make_shared<op::Subtract>();
                        sub_op->set_name(node_proto.output(0) + "_sub");
                        auto sub_gnode =
                            m_graph->add_node_and_edge(sub_op, {input_index, max_index});

                        // Exp
                        auto exp_op = std::make_shared<op::Exp>();
                        exp_op->set_name(node_proto.output(0) + "_exp");
                        auto exp_gnode = m_graph->add_node_and_edge(exp_op, {sub_gnode});

                        // Sum
                        auto sum_op = std::make_shared<op::Sum>(ng_axes_softmax);
                        sum_op->set_name(node_proto.output(0) + "_sum");
                        auto sum_gnode = m_graph->add_node_and_edge(sum_op, {exp_gnode});

                        // Reshape
                        reshape_op = std::make_shared<op::Reshape>(ng_axis_order, extend_max_shape);
                        auto sum_reshape_gnode =
                            m_graph->add_node_and_edge(reshape_op, {sum_gnode});

                        // Broadcast
                        auto exp_index = GNodeIndex(exp_gnode);
                        auto sum_index = GNodeIndex(sum_reshape_gnode);
                        std::tie(exp_index, sum_index) =
                            graph::numpy_broadcast(std::make_pair(exp_index, sum_index), m_graph);

                        // Div
                        auto div_op = std::make_shared<op::Divide>();
                        div_op->set_name(node_proto.output(0) + "_div");
                        auto div_gnode = m_graph->add_node_and_edge(div_op, {exp_index, sum_index});
                        NamedNodeVector ret{{node_proto.output(0), div_gnode}};
                        return ret;
                    }
                }

                NamedNodeVector
                    TranslateSoftmaxGradOp(const onnx::NodeProto& node_proto,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto dy_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto y_index = GetInputIndex(all_ng_nodes, node_proto, 1);
                    Node node(node_proto);
                    auto axis = node.get_attribute_value<std::vector<int64_t>>("axis", {-1})[0];
                    axis += axis < 0 ? dy_index.get_shape().size() : 0;
                    nnfusion::AxisSet ng_axes_softmax;
                    ng_axes_softmax.insert(axis);
                    auto softmax_grad_op = std::make_shared<op::SoftmaxGrad>(ng_axes_softmax);
                    softmax_grad_op->set_name(node_proto.output(0));
                    auto softmax_grad_gnode =
                        m_graph->add_node_and_edge(softmax_grad_op, {dy_index, y_index});
                    NamedNodeVector ret{{node_proto.output(0), softmax_grad_gnode}};
                    return ret;
                }

                NamedNodeVector TranslateSparseSoftmaxCrossEntropyOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto logits_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto label_index = GetInputIndex(all_ng_nodes, node_proto, 1);

                    auto logits_shape = logits_index.get_shape();
                    auto label_shape = label_index.get_shape();
                    auto sample_num = shape_size(label_shape);
                    Shape flatten_shape{sample_num, logits_shape.back()};
                    NNFUSION_CHECK(logits_shape.size() - label_shape.size() == 1)
                        << "SparseSoftmaxCrossEntropy should be (N+1)-D logits with N-D label, but "
                           "found "
                        << logits_index.get_shape().size() << "-D logits and "
                        << label_index.get_shape().size() << "-D label.";

                    ///\todo different weight sematic
                    // For SparseSoftmaxCrossEntropy, weight is sample weight, see https://github.com/microsoft/onnxruntime/blob/bd215b79a2aeeb5e0b77feaaed67cc7e3ac98ead/orttraining/orttraining/training_ops/cuda/loss/softmaxcrossentropy_impl.cc#L164
                    // For SoftmaxCrossEntropyLoss, weight is class weight, see: https://github.com/onnx/onnx/blob/master/docs/Operators.md#softmaxcrossentropyloss
                    std::shared_ptr<graph::GNode> sample_weight_gnode = nullptr;
                    if (node_proto.input_size() > 2)
                    {
                        auto weight_index = GetInputIndex(all_ng_nodes, node_proto, 2);
                        if (node_proto.op_type() == "SoftmaxCrossEntropyLoss")
                        {
                            NNFUSION_CHECK(weight_index.get_shape().size() == 1);

                            nnfusion::op::OpConfig::any myConfig;
                            myConfig["axis"] = 0;

                            auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                                node_proto.output(0), "GatherV2", myConfig);
                            sample_weight_gnode =
                                m_graph->add_node_and_edge(generic_op, {weight_index, label_index});
                        }
                        else if (node_proto.op_type() == "SparseSoftmaxCrossEntropy")
                        {
                            NNFUSION_CHECK(weight_index.get_shape() == label_shape);
                            NNFUSION_CHECK(weight_index.gnode->get_output_size() == 1);
                            sample_weight_gnode = weight_index.gnode;
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL() << "unsupported weight input in op "
                                                  << node_proto.op_type();
                        }
                    }
                    else
                    {
                        auto sample_weight_const =
                            std::make_shared<op::Constant>(logits_index.get_element_type(),
                                                           label_index.get_shape(),
                                                           std::vector<std::string>{"1.0"});
                        sample_weight_gnode =
                            m_graph->add_node_and_edge(sample_weight_const, GNodeIndexVector{});
                    }
                    if (sample_weight_gnode->get_shape().size() > 1)
                    {
                        auto weight_reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(label_shape.size()),
                            Shape{sample_num});
                        sample_weight_gnode =
                            m_graph->add_node_and_edge(weight_reshape_op, {sample_weight_gnode});
                    }

                    NamedNodeVector ret(2, {"", nullptr});

                    // softmax output1 is optional
                    string log_prob_name;
                    if (node_proto.output_size() > 1)
                    {
                        log_prob_name = node_proto.output(1);
                    }
                    else
                    {
                        log_prob_name = node_proto.name() + "_log_prob";
                    }

                    if (logits_shape.size() > 2)
                    {
                        auto logits_reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(logits_shape.size()), flatten_shape);
                        auto logits_reshape_gnode =
                            m_graph->add_node_and_edge(logits_reshape_op, {logits_index});
                        logits_index = GNodeIndex{logits_reshape_gnode, 0};

                        Shape softmax_label_shape{sample_num};
                        auto label_reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(label_shape.size()),
                            softmax_label_shape);
                        auto label_reshape_gnode =
                            m_graph->add_node_and_edge(label_reshape_op, {label_index});
                        label_index = GNodeIndex{label_reshape_gnode, 0};
                    }
                    Node node(node_proto);
                    auto reduction = node.get_attribute_value<std::string>("reduction", "mean");

                    nnfusion::AxisSet ng_axes_softmax{logits_index.get_shape().size() -
                                                      1}; // along the last dim
                    // auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                    auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax, true);
                    softmax_op->set_name(log_prob_name);
                    auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {logits_index});

                    std::shared_ptr<nnfusion::graph::GNode> log_prob = nullptr;
                    // reshape back
                    if (logits_shape.size() > 2)
                    {
                        auto softmax_reshape_back_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(softmax_gnode->get_shape().size()),
                            logits_shape);
                        auto softmax_reshape_back_gnode =
                            m_graph->add_node_and_edge(softmax_reshape_back_op, {softmax_gnode});
                        // log_prob = m_graph->add_node_and_edge(std::make_shared<op::Log>(),
                        //                                       {softmax_reshape_back_gnode});
                        ret[1] = NamedNode{log_prob_name, softmax_reshape_back_gnode};
                    }
                    else
                    {
                        // log_prob = m_graph->add_node_and_edge(std::make_shared<op::Log>(),
                        //                                       {softmax_gnode});
                        ret[1] = NamedNode{log_prob_name, softmax_gnode};
                    }
                    // ret[1] = NamedNode{log_prob_name, log_prob};
                    nnfusion::op::OpConfig::any ce_config;
                    ce_config["in_log_space"] = false;

                    auto loss_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0) + "_ce",
                        "CrossEntropyAvgLossWithLabels", // select which existing kernels to use;
                        ce_config);
                    auto loss_gnode = m_graph->add_node_and_edge(
                        loss_op, {GNodeIndex{softmax_gnode}, label_index});

                    loss_gnode = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                            {loss_gnode, sample_weight_gnode});

                    if (reduction == "mean")
                    {
                        auto loss_shape = loss_gnode->get_shape();
                        auto sum_gnode = m_graph->add_node_and_edge(
                            std::make_shared<op::Sum>(get_default_order(loss_shape)), {loss_gnode});
                        auto sum_weight_gnode =
                            m_graph->add_node_and_edge(std::make_shared<op::Sum>(get_default_order(
                                                           sample_weight_gnode->get_shape())),
                                                       {sample_weight_gnode});
                        auto mean_op = std::make_shared<op::Divide>();
                        mean_op->set_name(node_proto.output(0));
                        auto mean_gnode =
                            m_graph->add_node_and_edge(mean_op, {sum_gnode, sum_weight_gnode});
                        ret[0] = NamedNode{node_proto.output(0), mean_gnode};
                    }
                    else if (reduction == "sum")
                    {
                        auto sum_op =
                            std::make_shared<op::Sum>(get_default_order(loss_gnode->get_shape()));
                        sum_op->set_name(node_proto.output(0));
                        auto sum_gnode = m_graph->add_node_and_edge(sum_op, {loss_gnode});
                        ret[0] = NamedNode{node_proto.output(0), sum_gnode};
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "unsupported softmax cross entropy reduction: "
                                              << reduction;
                    }

                    return ret;
                }

                NamedNodeVector TranslateSparseSoftmaxCrossEntropyGradOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto loss_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto logits_index = GetInputIndex(all_ng_nodes, node_proto, 1);
                    auto label_index = GetInputIndex(all_ng_nodes, node_proto, 2);
                    // TODO: support weights
                    // GNodeIndex weight_index{nullptr};
                    // if (node_proto.input_size() > 3)
                    // {
                    //     weight_index = GetInputIndex(all_ng_nodes, node_proto, 3);
                    // }

                    // convert log_prob to logits
                    auto logits_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Exp>(), {logits_index});
                    logits_index = GNodeIndex{logits_gnode};

                    auto logits_shape = logits_index.get_shape();
                    auto label_shape = label_index.get_shape();
                    NNFUSION_CHECK(logits_shape.size() - label_shape.size() == 1)
                        << "SparseSoftmaxCrossEntropyGrad should be (N+1)-D logits with N-D label, "
                           "but "
                           "found "
                        << logits_index.get_shape().size() << "-D logits and "
                        << label_index.get_shape().size() << "-D label.";

                    if (logits_shape.size() > 2)
                    {
                        auto sample_num = shape_size(label_index.get_shape());
                        Shape softmax_logits_shape{sample_num, logits_shape.back()};
                        auto logits_reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(logits_index.get_shape().size()),
                            softmax_logits_shape);
                        auto logits_reshape_gnode =
                            m_graph->add_node_and_edge(logits_reshape_op, {logits_index});
                        logits_index = GNodeIndex{logits_reshape_gnode, 0};

                        Shape softmax_label_shape{sample_num};
                        auto label_reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(label_index.get_shape().size()),
                            softmax_label_shape);
                        auto label_reshape_gnode =
                            m_graph->add_node_and_edge(label_reshape_op, {label_index});
                        label_index = GNodeIndex{label_reshape_gnode, 0};
                    }

                    Node node(node_proto);
                    auto reduction = node.get_attribute_value<std::string>("reduction", "mean");

                    std::shared_ptr<nnfusion::graph::GNode> sample_loss = nullptr;
                    if (reduction == "mean")
                    {
                        size_t batch_size = nnfusion::shape_size(label_index.get_shape());
                        std::tie(loss_index, logits_index) =
                            numpy_broadcast(std::make_pair(loss_index, logits_index), m_graph);

                        const auto& et = loss_index.gnode->get_element_type();
                        auto divisor_const = std::make_shared<op::Constant>(
                            et,
                            logits_index.get_shape(),
                            std::vector<std::string>{std::to_string(batch_size)});
                        auto divisor_gnode =
                            m_graph->add_node_and_edge(divisor_const, GNodeIndexVector{});
                        sample_loss =
                            m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                       {loss_index, GNodeIndex{divisor_gnode}});
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "unsupported softmax cross entropy reduction: "
                                              << reduction;
                    }

                    // CrossEntropyFwdBwdWithSoftmaxBwd will computing grad with every output_grad of 1, so we should multiply sample_loss
                    auto bwd_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0) + "_cefwdbwd_softmaxbwd",
                        "CrossEntropyFwdBwdWithSoftmaxBwdLarge", // select which existing kernels to use;
                        nnfusion::op::OpConfig::any{});

                    auto bwd_gnode =
                        m_graph->add_node_and_edge(bwd_op, {logits_index, label_index});

                    auto logits_grad_op = std::make_shared<op::Multiply>();
                    logits_grad_op->set_name(node_proto.output(0));
                    auto logits_grad =
                        m_graph->add_node_and_edge(logits_grad_op, {sample_loss, bwd_gnode});

                    // reshape back
                    if (logits_shape.size() > 2)
                    {
                        auto logits_grad_reshape_back_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(logits_grad->get_shape().size()),
                            logits_shape);
                        auto logits_grad_reshape_back_gnode =
                            m_graph->add_node_and_edge(logits_grad_reshape_back_op, {logits_grad});
                        return {{node_proto.output(0), logits_grad_reshape_back_gnode}};
                    }
                    else
                    {
                        return {{node_proto.output(0), logits_grad}};
                    }
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion