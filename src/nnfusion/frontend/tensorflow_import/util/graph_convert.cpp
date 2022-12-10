//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include "graph_convert.hpp"
#include "../ops/const.hpp"
#include "nnfusion/common/axis_vector.hpp"
#include "nnfusion/common/coordinate_diff.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"

#include "nnfusion/frontend/tensorflow_import/util/bcast.hpp"

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fdefault_device);
DEFINE_bool(fantares_mode, false, "Enable antares mode.");
DEFINE_bool(fnchw, true, "Convert dataformat to nchw.");
// todo: add control edge ?
namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            // Using this policy if no explict tf_import mapping exists
            NamedNodeVector
                TranslateGenericNoAttrOp(const tensorflow::NodeDef& node,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes;
                size_t input_cnt = node.input_size();
                for (int i = 0; i < input_cnt; i++)
                    input_gnodes.push_back(GetInputNode(all_ng_nodes, node, i));

                nnfusion::op::OpConfig::any config;
                for (auto& entry : node.attr())
                {
                    switch (entry.second.value_case())
                    {
                    case ::tensorflow::AttrValue::ValueCase::kS:
                        config[entry.first] = entry.second.s();
                        break;
                    case ::tensorflow::AttrValue::ValueCase::kI:
                        config[entry.first] = entry.second.i();
                        break;
                    case ::tensorflow::AttrValue::ValueCase::kF:
                        config[entry.first] = entry.second.f();
                        break;
                    case ::tensorflow::AttrValue::ValueCase::kB:
                        config[entry.first] = entry.second.b();
                        break;
                    case ::tensorflow::AttrValue::ValueCase::kType:
                    {
                        auto dtype = entry.second.type();
                        switch (dtype)
                        {
                        case ::tensorflow::DataType::DT_FLOAT:
                            config[entry.first] = "float32";
                            break;
                        case ::tensorflow::DataType::DT_INT32: config[entry.first] = "int32"; break;
                        case ::tensorflow::DataType::DT_HALF:
                            config[entry.first] = "float16";
                            break;
                        default: NNFUSION_CHECK(false) << "Unrecognized data type: " << dtype;
                        }
                    }
                    break;
                    default:
                        NNFUSION_CHECK(false) << "Unrecognized value case: "
                                              << entry.second.value_case();
                    }
                }
                NNFUSION_LOG(INFO) << "GenericTFNode(" << node.op() << "): " << config << std::endl;

                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    node.name(),
                    node.op(), // select which existing kernels to use;
                    config);
                auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_gnodes);
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateIdentityOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_node = GetInputNode(all_ng_nodes, node, 0);
                NamedNodeVector ret{{node.name(), input_node}};
                return ret;
            }

            NamedNodeVector
                TranslateInvertPermutationOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnodes = GetAllInputNode(all_ng_nodes, node);
                nnfusion::op::OpConfig::any myConfig;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_gnodes);
                NamedNodeVector ret{{node.name(), generic_gnode}};

                return ret;
            }

            NamedNodeVector TranslateNoOp(const tensorflow::NodeDef& node,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                NamedNodeVector ret;
                size_t input_cnt = node.input_size();
                for (int i = 0; i < input_cnt; i++)
                {
                    TensorId input_tensor(ParseTensorName(node.input(i)));
                    if (input_tensor.second >= 0)
                    {
                        auto input_node = GetInputNode(all_ng_nodes, node, i);
                        ret.push_back({node.name(), input_node});
                    }
                }
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateUnaryOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto ng_node = std::make_shared<T>();
                ng_node->set_name(node.name());
                auto gnode = m_graph->add_node_and_edge(ng_node, {input_gnode});
                NamedNodeVector ret{{node.name(), gnode}};
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateBinaryOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::tie(lhs_gnode, rhs_gnode) =
                    graph::numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                auto op = std::make_shared<T>();
                op->set_name(node.name());
                auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});

                NamedNodeVector ret{{node.name(), gnode}};
                return ret;
            }

            template <typename T>
            NamedNodeVector TranslateTensorOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                tensorflow::DataType dtype;
                auto status = GetNodeAttr(node.attr(), "dtype", dtype);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et;
                status = TFDataTypeToNNFusionElementType(dtype, &nnfusion_et);
                NNFUSION_CHECK(status) << "DataType " << dtype << " is not supported.";
                tensorflow::TensorShapeProto tf_shape = node.attr().at("shape").shape();
                nnfusion::Shape ng_shape;
                status = TFTensorShapeToNGraphShape(tf_shape, &ng_shape);
                NNFUSION_CHECK(status);

                shared_ptr<T> input_op;
                if (node.op() == "VariableV2")
                {
                    input_op = std::make_shared<T>(nnfusion_et, ng_shape, false, true);
                }
                else
                {
                    input_op = std::make_shared<T>(nnfusion_et, ng_shape);
                }
                input_op->set_name(node.name());
                auto input_gnode = m_graph->add_node_and_edge(input_op, GNodeVector({}));
                NamedNodeVector ret{{node.name(), input_gnode}};
                return ret;
            }

            std::shared_ptr<nnfusion::graph::GNode>
                TranslateSoftmaxToBasicOp(const std::shared_ptr<nnfusion::graph::GNode> input_gnode,
                                          const nnfusion::AxisSet& reduction_axes,
                                          const std::string& node_name,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // softmax = exp(logits) / reduce_sum(exp(logits), axis).
                // Exp op.
                auto exp_op = std::make_shared<op::Exp>();
                auto exp_gnode = m_graph->add_node_and_edge(exp_op, {input_gnode});

                // Sum op with keepdims=true.
                auto sum_op = std::make_shared<op::Sum>(reduction_axes);
                auto sum_gnode = m_graph->add_node_and_edge(sum_op, {exp_gnode});

                nnfusion::Shape input_shape = input_gnode->get_shape();
                size_t input_rank = input_shape.size();
                nnfusion::Shape result_shape_with_keep(input_rank);

                for (size_t i = 0; i < input_rank; i++)
                {
                    result_shape_with_keep[i] = reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                }
                nnfusion::AxisVector axis_order(sum_gnode->get_shape().size());
                std::iota(axis_order.begin(), axis_order.end(), 0);
                auto reshape_op = std::make_shared<op::Reshape>(axis_order, result_shape_with_keep);
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {sum_gnode});

                // Divide op.
                std::tie(exp_gnode, reshape_gnode) =
                    graph::numpy_broadcast(std::make_pair(exp_gnode, reshape_gnode), m_graph);

                auto div_op = std::make_shared<op::Divide>();
                div_op->set_name(node_name);
                auto div_gnode = m_graph->add_node_and_edge(div_op, {exp_gnode, reshape_gnode});

                return div_gnode;
            }

            NamedNodeVector TranslateSparseSoftmaxCrossEntropyWithLogitsOp(
                const tensorflow::NodeDef& node,
                const NodeMap& all_ng_nodes,
                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);

                nnfusion::AxisSet ng_axes_softmax{lhs_gnode->get_shape().size() - 1};

                if (!FLAGS_fantares_mode)
                {
                    auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                    auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {lhs_gnode});

                    auto loss_op = std::make_shared<nnfusion::op::GenericOp>(
                        node.name(),
                        "CrossEntropyAvgLossWithLabels", // select which existing kernels to use;
                        nnfusion::op::OpConfig::any{});
                    auto loss_gnode =
                        m_graph->add_node_and_edge(loss_op, {softmax_gnode, rhs_gnode});

                    auto bwd_op = std::make_shared<nnfusion::op::GenericOp>(
                        node.name(),
                        "CrossEntropyFwdBwdWithSoftmaxBwd", // select which existing kernels to use;
                        nnfusion::op::OpConfig::any{});
                    auto bwd_gnode = m_graph->add_node_and_edge(bwd_op, {softmax_gnode, rhs_gnode});

                    NamedNodeVector ret{{node.name(), loss_gnode}, {node.name(), bwd_gnode}};
                    return ret;
                }
                else
                {
                    // auto softmax_gnode =
                    //     TranslateSoftmaxToBasicOp(lhs_gnode, ng_axes_softmax, node.name(), m_graph);
                    auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                    auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {lhs_gnode});
                    NNFUSION_CHECK(softmax_gnode->get_shape().size() == 2);
                    // OneHot op.
                    nnfusion::op::OpConfig::any onehot_config;
                    onehot_config["axis"] = -1;
                    onehot_config["depth"] = softmax_gnode->get_shape()[1];
                    onehot_config["off_value"] = 0;
                    onehot_config["on_value"] = 1;
                    onehot_config["T"] = "float";

                    auto onehot_op = std::make_shared<nnfusion::op::GenericOp>(
                        node.name(), "OneHot", onehot_config);

                    auto onehot_gnode = m_graph->add_node_and_edge(onehot_op, {rhs_gnode});

                    // Subtract op.
                    auto subtract_op = std::make_shared<op::Subtract>();
                    subtract_op->set_name(node.name());
                    auto bwd_gnode =
                        m_graph->add_node_and_edge(subtract_op, {softmax_gnode, onehot_gnode});

                    // Log op.
                    auto log_op = std::make_shared<op::Log>();
                    auto log_gnode = m_graph->add_node_and_edge(log_op, {softmax_gnode});

                    // Multiply op.
                    auto multiply_op = std::make_shared<op::Multiply>();
                    auto multiply_gnode =
                        m_graph->add_node_and_edge(multiply_op, {log_gnode, onehot_gnode});

                    // Negative op.
                    auto neg_op = std::make_shared<op::Negative>();
                    auto neg_gnode = m_graph->add_node_and_edge(neg_op, {multiply_gnode});

                    // Sum op.
                    nnfusion::AxisSet reduction_axes;
                    reduction_axes.insert(neg_gnode->get_shape().size() - 1);
                    auto sum_op = std::make_shared<op::Sum>(reduction_axes);
                    sum_op->set_name(node.name());
                    auto loss_gnode = m_graph->add_node_and_edge(sum_op, {neg_gnode});

                    NamedNodeVector ret{{node.name(), loss_gnode}, {node.name(), bwd_gnode}};
                    return ret;
                }
            }

            NamedNodeVector TranslateMatMulOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);
                // Transpose arguments if requested.
                bool transpose_a = false;
                bool transpose_b = false;
                bool status = GetNodeAttr(node.attr(), "transpose_a", transpose_a);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "transpose_b", transpose_b);
                NNFUSION_CHECK(status);
                // if (transpose_a)
                // {
                //     ng_lhs = nnfusion::graph::numpy_transpose(ng_lhs, nnfusion::AxisVector{1, 0});
                // }
                // if (transpose_b)
                // {
                //     ng_rhs = nnfusion::graph::numpy_transpose(ng_rhs, nnfusion::AxisVector{1, 0});
                // }

                auto dot_op =
                    std::make_shared<nnfusion::op::Dot>(0, false, transpose_a, transpose_b);
                //ng_node->set_transpose(transpose_a, transpose_b);

                dot_op->set_name(node.name());
                auto dot_gnode = m_graph->add_node_and_edge(dot_op, {lhs_gnode, rhs_gnode});

                NamedNodeVector ret{{node.name(), dot_gnode}};
                return ret;
            }

            NamedNodeVector TranslateBatchMatMulOp(const tensorflow::NodeDef& node,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);
                // Transpose arguments if requested.
                bool adj_x = false;
                bool adj_y = false;

                bool status = GetNodeAttr(node.attr(), "adj_x", adj_x);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "adj_y", adj_y);
                NNFUSION_CHECK(status);

                int input_dims = lhs_gnode->get_output_shape(0).size();

                nnfusion::op::OpConfig::any myConfig;
                myConfig["adj_x"]["b"] = adj_x;
                myConfig["adj_y"]["b"] = adj_y;

                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    node.name(),
                    "BatchMatMul", // select which existing kernels to use;
                    myConfig);
                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {lhs_gnode, rhs_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateBiasAddOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto bias_gnode = GetInputNode(all_ng_nodes, node, 1);
                std::string tf_data_format;
                bool status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "BiasAdd data format is neither NHWC nor NCHW";

                auto input_shape = input_gnode->get_shape();
                auto bias_shape = bias_gnode->get_shape();

                NNFUSION_CHECK(bias_shape.size() == 1)
                    << "Bias argument to BiasAdd does not have one dimension";

                bool is_nhwc = (tf_data_format == "NHWC");

                nnfusion::AxisSet broadcast_axes;

                if (is_nhwc)
                {
                    for (size_t i = 0; i < input_shape.size() - 1; i++)
                    {
                        broadcast_axes.insert(i);
                    }
                }
                else
                {
                    for (size_t i = 0; i < input_shape.size(); i++)
                    {
                        if (i != 1)
                        {
                            broadcast_axes.insert(i);
                        }
                    }
                }

                auto bias_broadcasted_op =
                    std::make_shared<op::Broadcast>(input_shape, broadcast_axes);

                auto bias_broadcasted_gnode =
                    m_graph->add_node_and_edge(bias_broadcasted_op, {bias_gnode});

                auto add_op = std::make_shared<op::Add>();
                add_op->set_name(node.name());

                auto add_gnode =
                    m_graph->add_node_and_edge(add_op, {input_gnode, bias_broadcasted_gnode});

                NamedNodeVector ret{{node.name(), add_gnode}};
                return ret;
            }

            NamedNodeVector TranslateReluGradOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto arg_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto relu_grad_op = std::make_shared<op::ReluBackprop>();
                relu_grad_op->set_name(node.name());

                auto relu_grad_gnode =
                    m_graph->add_node_and_edge(relu_grad_op, {arg_gnode, delta_gnode});
                NamedNodeVector ret{{node.name(), relu_grad_gnode}};
                return ret;
            }

            NamedNodeVector TranslateRelu6GradOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto arg_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto relu_grad_op = std::make_shared<op::Relu6Backprop>();
                relu_grad_op->set_name(node.name());

                auto relu_grad_gnode =
                    m_graph->add_node_and_edge(relu_grad_op, {arg_gnode, delta_gnode});
                NamedNodeVector ret{{node.name(), relu_grad_gnode}};
                return ret;
            }

            NamedNodeVector TranslateSigmoidGradOp(const tensorflow::NodeDef& node,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto x0 = GetInputNode(all_ng_nodes, node, 0);
                auto x1 = GetInputNode(all_ng_nodes, node, 1);
                auto sigmoid_grad_op = std::make_shared<op::SigmoidBackprop>();
                sigmoid_grad_op->set_name(node.name());

                auto sigmoid_grad_gnode = m_graph->add_node_and_edge(sigmoid_grad_op, {x0, x1});
                NamedNodeVector ret{{node.name(), sigmoid_grad_gnode}};
                return ret;
            }

            NamedNodeVector TranslateBiasAddGradOp(const tensorflow::NodeDef& node,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                std::string tf_data_format;
                bool status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                if (tf_data_format == "")
                {
                    tf_data_format = "NHWC";
                }

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "BiasAddGrad data format is neither NHWC nor NCHW";

                auto input_shape = input_gnode->get_shape();

                NNFUSION_CHECK(input_shape.size() >= 2) << "Input tensor must be at least 2D";

                bool is_nhwc = (tf_data_format == "NHWC");

                nnfusion::AxisSet ng_reduction_axes;

                if (is_nhwc)
                {
                    for (size_t i = 0; i < input_shape.size() - 1; i++)
                    {
                        ng_reduction_axes.insert(i);
                    }
                }

                else
                {
                    for (size_t i = 0; i < input_shape.size(); i++)
                    {
                        if (i != 1)
                        {
                            ng_reduction_axes.insert(i);
                        }
                    }
                }

                auto bias_add_grad_op = std::make_shared<op::Sum>(ng_reduction_axes);
                bias_add_grad_op->set_name(node.name());
                auto bias_add_grad_gnode =
                    m_graph->add_node_and_edge(bias_add_grad_op, GNodeVector({input_gnode}));

                NamedNodeVector ret{{node.name(), bias_add_grad_gnode}};
                return ret;
            }

            NamedNodeVector TranslateReshapeOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto shape_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> shape;
                bool status = GetValueFromNGraphOp<int64>(shape_gnode, &shape);
                NNFUSION_CHECK(status);

                size_t output_rank = shape.size();
                size_t num_input_elements = nnfusion::shape_size(input_gnode->get_shape());

                // If there is a single "-1" in the result shape, we have to auto-infer
                // the length of that dimension.
                size_t inferred_pos;
                size_t product_of_rest = 1;
                bool seen_inferred = false;
                for (size_t i = 0; i < output_rank; i++)
                {
                    if (shape[i] == -1)
                    {
                        NNFUSION_CHECK(!seen_inferred);
                        //if (seen_inferred)
                        //{
                        //    return errors::InvalidArgument("Multiple -1 dimensions in result shape");
                        //}
                        inferred_pos = i;
                        seen_inferred = true;
                    }
                    else
                    {
                        product_of_rest *= shape[i];
                    }
                }
                if (seen_inferred)
                {
                    /*
                    if (num_input_elements % product_of_rest != 0)
                    {
                        NGRAPH_VLOG(3) << "{" << ng::join(input_gnode->get_shape()) << "}";
                        NGRAPH_VLOG(3) << "{" << ng::join(shape) << "}";
                        return errors::InvalidArgument(
                            "Product of known dimensions (", product_of_rest,
                            ") does not evenly divide the number of input elements (",
                            num_input_elements, ")");
                    }
                    */
                    NNFUSION_CHECK(num_input_elements % product_of_rest == 0);
                    shape[inferred_pos] = num_input_elements / product_of_rest;
                }

                // Convert the values from the constant into an nnfusion::Shape, and
                // construct the axis order while we are at it.
                nnfusion::Shape ng_shape(output_rank);

                for (size_t i = 0; i < output_rank; i++)
                {
                    ng_shape[i] = shape[i];
                }

                nnfusion::AxisVector ng_axis_order(input_gnode->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto reshape_op = std::make_shared<nnfusion::op::Reshape>(ng_axis_order, ng_shape);
                reshape_op->set_name(node.name());
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_gnode});

                NamedNodeVector ret{{node.name(), reshape_gnode}};
                return ret;
            }

            NamedNodeVector TranslateCastOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                tensorflow::DataType dtype;
                bool status = GetNodeAttr(node.attr(), "DstT", dtype);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et;
                status = TFDataTypeToNNFusionElementType(dtype, &nnfusion_et);
                NNFUSION_CHECK(status);
                auto cast_op = std::make_shared<op::Convert>(nnfusion_et);
                cast_op->set_name(node.name());
                auto cast_gnode = m_graph->add_node_and_edge(cast_op, {input_gnode});

                NamedNodeVector ret{{node.name(), cast_gnode}};
                return ret;
            }

            NamedNodeVector TranslateMaxPoolOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                std::vector<int32> tf_strides;
                std::vector<int32> tf_ksize;
                std::string tf_padding_type;
                std::string tf_data_format;

                bool status;
                status = GetNodeAttr(node.attr(), "strides", tf_strides);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "ksize", tf_ksize);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "padding", tf_padding_type);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "MaxPool data format is neither NHWC nor NCHW";

                bool is_nhwc = (tf_data_format == "NHWC");
                nnfusion::Strides ng_strides(2);
                nnfusion::Shape ng_image_shape(2);
                nnfusion::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, input_gnode->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
                auto device = get_device_type(FLAGS_fdefault_device);
                std::shared_ptr<GNode> reshape_gnode;
                if (is_nhwc && FLAGS_fnchw)
                {
                    reshape_gnode = BatchToNGraph(is_nhwc, input_gnode);
                    NNFUSION_CHECK_NOT_NULLPTR(reshape_gnode);
                    // Set data format as "NCHW", since have transposed the data from "NHWC" to "NCHW".
                    tf_data_format = "NCHW";
                    m_graph->add_node(reshape_gnode);
                    m_graph->add_edge(input_gnode, 0, reshape_gnode, 0);
                }
                else
                {
                    reshape_gnode = input_gnode;
                }

                // TODO: change this once nGraph supports negative padding
                // (CoordinateDiff) for MaxPool
                // ng::CoordinateDiff ng_padding_below{0,0};
                // ng::CoordinateDiff ng_padding_above{0,0};

                nnfusion::Shape ng_padding_below{0, 0};
                nnfusion::Shape ng_padding_above{0, 0};
                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);

                auto maxpool_op = std::make_shared<nnfusion::op::MaxPool>(ng_kernel_shape,
                                                                          ng_strides,
                                                                          ng_padding_below,
                                                                          ng_padding_above,
                                                                          tf_data_format);
                auto maxpool_gnode = m_graph->add_node_and_edge(maxpool_op, {reshape_gnode});
                std::shared_ptr<GNode> reshape_maxpool_gnode;
                if (is_nhwc && FLAGS_fnchw)
                {
                    reshape_maxpool_gnode = BatchToTensorflow(is_nhwc, maxpool_gnode);
                    NNFUSION_CHECK_NOT_NULLPTR(reshape_maxpool_gnode);
                    m_graph->add_node(reshape_maxpool_gnode);
                    m_graph->add_edge(maxpool_gnode, 0, reshape_maxpool_gnode, 0);
                }
                else
                {
                    reshape_maxpool_gnode = maxpool_gnode;
                }
                reshape_maxpool_gnode->get_op_ptr()->set_name(node.name());
                reshape_maxpool_gnode->set_name(node.name());

                NamedNodeVector ret{{node.name(), reshape_maxpool_gnode}};
                return ret;
            }

            NamedNodeVector TranslateConv2DOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // <Todo: wenxh> Group Conv2D
                std::vector<int> tf_strides;
                std::vector<int> tf_dilations;
                std::string tf_padding_type;
                std::string tf_data_format;
                // Make sure the order maters!
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto filter_gnode = GetInputNode(all_ng_nodes, node, 1);

                bool status;
                status = GetNodeAttr(node.attr(), "strides", tf_strides);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "dilations", tf_dilations);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "padding", tf_padding_type);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "Conv2D data format is neither NHWC nor NCHW";
                bool is_nhwc = (tf_data_format == "NHWC");
                nnfusion::Strides ng_strides(2);
                nnfusion::Strides ng_dilations(2);
                nnfusion::Shape ng_image_shape(2);
                nnfusion::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, input_gnode->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);

                std::shared_ptr<GNode> reshape_input_gnode;
                if (FLAGS_fnchw && is_nhwc)
                {
                    reshape_input_gnode = BatchToNGraph(is_nhwc, input_gnode);
                    NNFUSION_CHECK_NOT_NULLPTR(reshape_input_gnode);
                    m_graph->add_node(reshape_input_gnode);
                    m_graph->add_edge(input_gnode, 0, reshape_input_gnode, 0);
                }
                else
                {
                    reshape_input_gnode = input_gnode;
                }

                auto& filter_shape = filter_gnode->get_shape();
                ng_kernel_shape[0] = filter_shape[0];
                ng_kernel_shape[1] = filter_shape[1];
                auto reshape_filter_gnode = Reshape<3, 2, 0, 1>(filter_gnode);
                if (!is_nhwc || FLAGS_fnchw)
                {
                    // Set data format as "NCHW", since have transposed the data from "NHWC" to "NCHW".
                    tf_data_format = "NCHW";
                    m_graph->add_node(reshape_filter_gnode);
                    m_graph->add_edge(filter_gnode, 0, reshape_filter_gnode, 0);
                }
                else
                {
                    reshape_filter_gnode = filter_gnode;
                }

                // Padding
                nnfusion::CoordinateDiff ng_padding_below{0, 0};
                nnfusion::CoordinateDiff ng_padding_above{0, 0};

                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_dilations,
                            ng_padding_below,
                            ng_padding_above);

                // NNFUSION_CHECK(ng_padding_below == ng_padding_above)
                //     << "Asymetric padding is not supported by now.";
                // Add a Pad op to avoid asymetric padding
                if (ng_padding_below != ng_padding_above)
                {
                    NNFUSION_CHECK(input_gnode->get_shape().size() == 4);
                    nnfusion::Shape padding_below(4, 0);
                    nnfusion::Shape padding_above(4, 0);
                    nnfusion::Shape padding_interior(4, 0);
                    is_nhwc = (tf_data_format == "NHWC");

                    for (size_t i = 0; i < 2; i++)
                    {
                        padding_below[i + (is_nhwc ? 1 : 2)] = ng_padding_below[i];
                        padding_above[i + (is_nhwc ? 1 : 2)] = ng_padding_above[i];
                        ng_padding_below[i] = 0;
                        ng_padding_above[i] = 0;
                    }

                    auto pad_val_op =
                        std::make_shared<op::Constant>(reshape_input_gnode->get_element_type(),
                                                       nnfusion::Shape{},
                                                       std::vector<std::string>{"0"});
                    auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                    auto pad_op =
                        std::make_shared<op::Pad>(padding_below, padding_above, padding_interior);
                    pad_op->set_name(node.name() + "Pad");

                    auto pad_gnode =
                        m_graph->add_node_and_edge(pad_op, {reshape_input_gnode, pad_val_gnode});
                    reshape_input_gnode = pad_gnode;
                }
                // Generate new op
                auto conv_op = std::make_shared<op::Convolution>(
                    ng_strides, ng_dilations, ng_padding_below, ng_padding_above, tf_data_format);
                auto conv_gnode = m_graph->add_node_and_edge(
                    conv_op, {reshape_input_gnode, reshape_filter_gnode});

                std::shared_ptr<GNode> reshape_conv_gnode;
                if (FLAGS_fnchw && is_nhwc)
                {
                    reshape_conv_gnode = BatchToTensorflow(is_nhwc, conv_gnode);
                    NNFUSION_CHECK_NOT_NULLPTR(reshape_conv_gnode);
                    m_graph->add_node(reshape_conv_gnode);
                    m_graph->add_edge(conv_gnode, 0, reshape_conv_gnode, 0);
                }
                else
                {
                    reshape_conv_gnode = conv_gnode;
                }
                reshape_conv_gnode->get_op_ptr()->set_name(node.name());
                reshape_conv_gnode->set_name(node.name());
                NamedNodeVector ret{{node.name(), reshape_conv_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateDepthwiseConv2dNativeOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto filter_gnode = GetInputNode(all_ng_nodes, node, 1);
                std::vector<int32> tf_strides;
                std::vector<int32> tf_dilations;
                std::string tf_padding_type;
                std::string tf_data_format;

                bool status;
                status = GetNodeAttr(node.attr(), "strides", tf_strides);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "dilations", tf_dilations);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "padding", tf_padding_type);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "Conv2D data format is neither NHWC nor NCHW";
                bool is_nhwc = (tf_data_format == "NHWC");

                Strides ng_strides(2);
                Strides ng_dilations(2);
                Shape ng_image_shape(2);
                Shape ng_kernel_shape(2);

                auto& filter_shape = filter_gnode->get_shape();
                ng_kernel_shape[0] = filter_shape[0];
                ng_kernel_shape[1] = filter_shape[1];

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, input_gnode->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);

                // auto reshape_input_gnode = BatchToNGraph(is_nhwc, input_gnode);
                // auto default_device = get_device_type(FLAGS_fdefault_device);
                // if (reshape_input_gnode != nullptr && default_device != GENERIC_CPU &&
                //     default_device != HLSL)
                // {
                //     m_graph->add_node(reshape_input_gnode);
                //     m_graph->add_edge(input_gnode, 0, reshape_input_gnode, 0);
                // }
                // else
                // {
                //     reshape_input_gnode = input_gnode;
                // }

                CoordinateDiff ng_padding_below{0, 0};
                CoordinateDiff ng_padding_above{0, 0};
                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_dilations,
                            ng_padding_below,
                            ng_padding_above);

                // NNFUSION_CHECK(ng_padding_below == ng_padding_above)
                //     << "Asymetric padding is not supported by now.";
                // Add a Pad op to avoid asymetric padding
                if (ng_padding_below != ng_padding_above)
                {
                    NNFUSION_CHECK(input_gnode->get_shape().size() == 4);
                    nnfusion::Shape padding_below(4, 0);
                    nnfusion::Shape padding_above(4, 0);
                    nnfusion::Shape padding_interior(4, 0);

                    for (size_t i = 0; i < 2; i++)
                    {
                        padding_below[i + (is_nhwc ? 1 : 2)] = ng_padding_below[i];
                        padding_above[i + (is_nhwc ? 1 : 2)] = ng_padding_above[i];
                        ng_padding_below[i] = 0;
                        ng_padding_above[i] = 0;
                    }

                    auto pad_val_op =
                        std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                       nnfusion::Shape{},
                                                       std::vector<std::string>{"0"});
                    auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                    auto pad_op =
                        std::make_shared<op::Pad>(padding_below, padding_above, padding_interior);
                    pad_op->set_name(node.name() + "Pad");

                    auto pad_gnode =
                        m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});
                    input_gnode = pad_gnode;
                }

                nnfusion::op::OpConfig::any op_config;
                op_config["data_format"] = tf_data_format;
                op_config["padding_type"] = tf_padding_type;
                op_config["strides"] = ng_strides;
                op_config["dilations"] = ng_dilations;
                op_config["padding_before"] = ng_padding_below;
                op_config["padding_after"] = ng_padding_above;

                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    node.name(), "DepthwiseConv2dNative", op_config);

                auto conv_gnode =
                    m_graph->add_node_and_edge(generic_op, {input_gnode, filter_gnode});

                // auto reshape_conv_gnode = BatchToTensorflow(is_nhwc, conv_gnode);
                // if (reshape_conv_gnode != nullptr && default_device != GENERIC_CPU &&
                //     default_device != HLSL)
                // {
                //     m_graph->add_node(reshape_conv_gnode);
                //     m_graph->add_edge(conv_gnode, 0, reshape_conv_gnode, 0);
                // }
                // else
                // {
                //     reshape_conv_gnode = conv_gnode;
                // }
                // reshape_conv_gnode->get_op_ptr()->set_name(node.name());
                conv_gnode->set_name(node.name());
                NamedNodeVector ret{{node.name(), conv_gnode}};
                return ret;
            }

            NamedNodeVector TranslateAvgPoolOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                std::vector<int32> tf_strides;
                std::vector<int32> tf_ksize;
                std::string tf_padding_type;
                std::string tf_data_format;

                bool status;
                status = GetNodeAttr(node.attr(), "strides", tf_strides);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "ksize", tf_ksize);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "padding", tf_padding_type);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "AvgPool data format is neither NHWC nor NCHW";

                bool is_nhwc = (tf_data_format == "NHWC");
                nnfusion::Strides ng_strides(2);
                nnfusion::Shape ng_image_shape(2);
                nnfusion::Shape ng_kernel_shape(2);

                BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
                BatchedOpParamToNGraph(is_nhwc, input_gnode->get_shape(), ng_image_shape);
                BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);

                auto reshape_gnode = BatchToNGraph(is_nhwc, input_gnode);
                if (reshape_gnode != nullptr)
                {
                    m_graph->add_node(reshape_gnode);
                    m_graph->add_edge(input_gnode, 0, reshape_gnode, 0);
                }
                else
                {
                    reshape_gnode = input_gnode;
                }

                // TODO: change this once nGraph supports negative padding
                // (CoordinateDiff) for AvgPool
                // ng::CoordinateDiff ng_padding_below{0,0};
                // ng::CoordinateDiff ng_padding_above{0,0};

                nnfusion::Shape ng_padding_below{0, 0};
                nnfusion::Shape ng_padding_above{0, 0};
                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);

                auto avgpool_op = std::make_shared<op::AvgPool>(
                    ng_kernel_shape, ng_strides, ng_padding_below, ng_padding_above, false);
                auto avgpool_gnode = m_graph->add_node_and_edge(avgpool_op, {reshape_gnode});

                auto reshape_avgpool_gnode = BatchToTensorflow(is_nhwc, avgpool_gnode);
                if (reshape_avgpool_gnode != nullptr)
                {
                    m_graph->add_node(reshape_avgpool_gnode);
                    m_graph->add_edge(avgpool_gnode, 0, reshape_avgpool_gnode, 0);
                }
                else
                {
                    reshape_avgpool_gnode = avgpool_gnode;
                }
                reshape_avgpool_gnode->get_op_ptr()->set_name(node.name());
                reshape_avgpool_gnode->set_name(node.name());

                NamedNodeVector ret{{node.name(), reshape_avgpool_gnode}};
                return ret;
            }

            NamedNodeVector TranslateFillOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto shape_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto value_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<size_t> dims_vec;
                bool status = GetValueFromNGraphOp<size_t>(shape_gnode, &dims_vec);
                NNFUSION_CHECK(status);

                nnfusion::Shape ng_output_shape(dims_vec.size());
                nnfusion::AxisSet ng_axis_set;
                for (size_t i = 0; i < dims_vec.size(); ++i)
                {
                    ng_output_shape[i] = dims_vec[i];
                    ng_axis_set.insert(i);
                }

                auto fill_op = std::make_shared<op::Broadcast>(ng_output_shape, ng_axis_set);
                fill_op->set_name(node.name());
                auto fill_gnode = m_graph->add_node_and_edge(fill_op, {value_gnode});

                NamedNodeVector ret{{node.name(), fill_gnode}};
                return ret;
            }

            NamedNodeVector TranslatePadOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto padding_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> paddings;
                bool status = GetValueFromNGraphOp<int64>(padding_gnode, &paddings);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(paddings.size() % 2 == 0)
                    << "Constant node for paddings does not have an even number of elements";

                nnfusion::Shape padding_below(paddings.size() / 2);
                nnfusion::Shape padding_above(paddings.size() / 2);
                nnfusion::Shape padding_interior(paddings.size() / 2);

                for (size_t i = 0; i < paddings.size() / 2; i++)
                {
                    padding_below[i] = paddings[2 * i];
                    padding_above[i] = paddings[2 * i + 1];
                    padding_interior[i] = 0;
                }

                // For PadV1 it seems the value is always zero.
                auto pad_val_op = std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                                 nnfusion::Shape{},
                                                                 std::vector<std::string>{"0"});
                auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                auto pad_op =
                    std::make_shared<op::Pad>(padding_below, padding_above, padding_interior);
                pad_op->set_name(node.name());

                auto pad_gnode = m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});

                NamedNodeVector ret{{node.name(), pad_gnode}};
                return ret;
            }

            NamedNodeVector TranslatePadV2Op(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto padding_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto constant_value_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> paddings;
                bool status = GetValueFromNGraphOp<int64>(padding_gnode, &paddings);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(constant_value_gnode->is_constant());
                auto constant_value_op =
                    std::dynamic_pointer_cast<op::Constant>(constant_value_gnode->get_op_ptr());
                auto constant_values = constant_value_op->get_value_strings();

                NNFUSION_CHECK(paddings.size() % 2 == 0)
                    << "Constant node for paddings does not have an even number of elements";

                nnfusion::Shape padding_below(paddings.size() / 2);
                nnfusion::Shape padding_above(paddings.size() / 2);
                nnfusion::Shape padding_interior(paddings.size() / 2);

                for (size_t i = 0; i < paddings.size() / 2; i++)
                {
                    padding_below[i] = paddings[2 * i];
                    padding_above[i] = paddings[2 * i + 1];
                    padding_interior[i] = 0;
                }

                auto pad_val_op = std::make_shared<op::Constant>(
                    input_gnode->get_element_type(), nnfusion::Shape{}, constant_values);

                auto pad_val_gnode = m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                auto pad_op =
                    std::make_shared<op::Pad>(padding_below, padding_above, padding_interior);
                pad_op->set_name(node.name());

                auto pad_gnode = m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});
                NamedNodeVector ret{{node.name(), pad_gnode}};
                return ret;
            }

            NamedNodeVector TranslateSpaceToDepthOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);

                std::string tf_data_format;
                int tf_block_size;
                bool status;
                status = GetNodeAttr(node.attr(), "block_size", tf_block_size);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["block_size"] = tf_block_size;
                myConfig["data_format"] = tf_data_format;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateFusedBatchNormOp(const tensorflow::NodeDef& node,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                bool tf_is_training;
                if (!GetNodeAttr(node.attr(), "is_training", tf_is_training))
                {
                    NNFUSION_LOG(INFO) << "is_training attribute not present, setting to true";
                    tf_is_training = true;
                }
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto scale_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto offset_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto mean_gnode = GetInputNode(all_ng_nodes, node, 3);
                auto variance_gnode = GetInputNode(all_ng_nodes, node, 4);

                std::string tf_data_format;
                bool status = GetNodeAttr(node.attr(), "data_format", tf_data_format);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW")
                    << "FusedBatchNorm data format is neither NHWC nor NCHW";

                bool is_nhwc = (tf_data_format == "NHWC");
                float tf_epsilon;
                if (!GetNodeAttr(node.attr(), "epsilon", tf_epsilon))
                {
                    NNFUSION_LOG(INFO) << "epsilon attribute not present, setting to 0.0001";
                    // TensorFlow default
                    tf_epsilon = 0.0001;
                }

                auto reshape_gnode = BatchToNGraph(is_nhwc, input_gnode);
                if (reshape_gnode != nullptr)
                {
                    m_graph->add_node(reshape_gnode);
                    m_graph->add_edge(input_gnode, 0, reshape_gnode, 0);
                }
                else
                {
                    reshape_gnode = input_gnode;
                }

                auto batch_norm_op = std::make_shared<op::BatchNormInference>(tf_epsilon);
                auto batch_norm_gnode = m_graph->add_node_and_edge(
                    batch_norm_op,
                    {scale_gnode, offset_gnode, reshape_gnode, mean_gnode, variance_gnode});
                auto reshape_batch_norm_gnode = BatchToTensorflow(is_nhwc, batch_norm_gnode);

                if (reshape_batch_norm_gnode != nullptr)
                {
                    m_graph->add_node(reshape_batch_norm_gnode);
                    m_graph->add_edge(batch_norm_gnode, 0, reshape_batch_norm_gnode, 0);
                }
                else
                {
                    reshape_batch_norm_gnode = batch_norm_gnode;
                }
                reshape_batch_norm_gnode->get_op_ptr()->set_name(node.name());
                reshape_batch_norm_gnode->set_name(node.name());

                NamedNodeVector ret{{node.name(), reshape_batch_norm_gnode}};

                return ret;
            }

            NamedNodeVector TranslateConcatV2Op(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                const int input_cnt = node.input_size();
                NNFUSION_CHECK(input_cnt >= 3) << "\"" << node.name()
                                               << "\" requires at least 3 inputs, got " << input_cnt
                                               << " instead";

                int max_inputs = INT_MAX - 1;
                if (get_device_type(FLAGS_fdefault_device) == ROCM_GPU)
                    max_inputs = 60;

                std::vector<GNodeVector> group_gnodes;
                GNodeVector arg_gnodes;
                for (int i = 0; i < input_cnt - 1; i++)
                {
                    auto arg_gnode = GetInputNode(all_ng_nodes, node, i);
                    arg_gnodes.push_back(arg_gnode);
                    if (arg_gnodes.size() == max_inputs)
                    {
                        group_gnodes.emplace_back(arg_gnodes);
                        arg_gnodes.clear();
                    }
                }
                if (arg_gnodes.size() > 0)
                    group_gnodes.emplace_back(std::move(arg_gnodes));
                NNFUSION_CHECK(group_gnodes.size() <= max_inputs);

                auto concat_axis_gnode = GetInputNode(all_ng_nodes, node, input_cnt - 1);
                std::vector<int> tf_concat_axis_vec;
                bool status = GetValueFromNGraphOp<int>(concat_axis_gnode, &tf_concat_axis_vec);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(tf_concat_axis_vec.size() == 1);

                int64 concat_axis = tf_concat_axis_vec[0];

                if (concat_axis < 0)
                {
                    concat_axis += int64(arg_gnodes[0]->get_shape().size());
                }

                GNodeVector merged_gnodes;
                for (int i = 0; i < group_gnodes.size(); ++i)
                {
                    auto concat_op = std::make_shared<nnfusion::op::Concat>(size_t(concat_axis));
                    concat_op->set_name(node.name() + "_" + std::to_string(i));
                    merged_gnodes.emplace_back(
                        m_graph->add_node_and_edge(concat_op, group_gnodes[i]));
                }
                if (merged_gnodes.size() > 1)
                {
                    auto concat_op = std::make_shared<nnfusion::op::Concat>(size_t(concat_axis));
                    concat_op->set_name(node.name() + "_merged");
                    auto next_gnode = m_graph->add_node_and_edge(concat_op, merged_gnodes);
                    merged_gnodes.clear();
                    merged_gnodes.emplace_back(next_gnode);
                }

                NamedNodeVector ret{{node.name(), merged_gnodes.front()}};
                return ret;
            }

            NamedNodeVector TranslateSumOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto axes_gnode = GetInputNode(all_ng_nodes, node, 1);

                bool tf_keep_dims;
                if (GetNodeAttr(node.attr(), "keep_dims", tf_keep_dims) == false)
                {
                    if (GetNodeAttr(node.attr(), "keepdims", tf_keep_dims) == false)
                    {
                        tf_keep_dims = false;
                    }
                }

                std::vector<int64> sum_axes;
                bool status = GetValueFromNGraphOp<int64>(axes_gnode, &sum_axes);
                NNFUSION_CHECK(status);

                nnfusion::Shape input_shape = input_gnode->get_shape();
                size_t input_rank = input_shape.size();

                status = CheckAxisDimInRange(sum_axes, input_rank);
                NNFUSION_CHECK(status);

                std::vector<size_t> ng_reduction_axes_vect(sum_axes.size());
                std::transform(
                    sum_axes.begin(),
                    sum_axes.end(),
                    ng_reduction_axes_vect.begin(),
                    [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
                nnfusion::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

                auto sum_op = std::make_shared<op::Sum>(ng_reduction_axes);
                NamedNodeVector ret;
                // If keep_dims is specified we need to reshape to put back the reduced
                // axes, with length 1.
                if (tf_keep_dims)
                {
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_gnode});
                    nnfusion::Shape ng_result_shape_with_keep(input_rank);

                    for (size_t i = 0; i < input_rank; i++)
                    {
                        ng_result_shape_with_keep[i] =
                            ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                    }
                    nnfusion::AxisVector ng_axis_order(sum_gnode->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto reshape_op =
                        std::make_shared<op::Reshape>(ng_axis_order, ng_result_shape_with_keep);
                    reshape_op->set_name(node.name());
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {sum_gnode});
                    ret.push_back({node.name(), reshape_gnode});
                }
                else
                {
                    sum_op->set_name(node.name());
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_gnode});
                    ret.push_back({node.name(), sum_gnode});
                }

                return ret;
            }

            NamedNodeVector TranslateSplitOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto split_dim_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto input_gnode = GetInputNode(all_ng_nodes, node, 1);

                // num_split : The number of ways to split. Must evenly divide
                // value.shape[split_dim]
                int32 num_split;
                bool status = GetNodeAttr(node.attr(), "num_split", num_split);
                NNFUSION_CHECK(status);
                nnfusion::Shape shape = input_gnode->get_shape();
                int rank = shape.size();
                std::vector<size_t> lower;
                std::vector<size_t> upper;
                for (int i = 0; i < rank; ++i)
                {
                    lower.push_back(0);
                    upper.push_back(shape[i]);
                }
                std::vector<int> split_dim_vec;
                status = GetValueFromNGraphOp<int>(split_dim_gnode, &split_dim_vec);
                NNFUSION_CHECK(status);
                int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);
                int size = shape[split_dim] / num_split;
                int cursor = 0;

                NamedNodeVector ret;

                for (size_t i = 0; i < num_split; ++i)
                {
                    lower[split_dim] = cursor;
                    cursor += size;
                    upper[split_dim] = cursor;
                    auto split_op = std::make_shared<op::Slice>(lower, upper);
                    //if (i > 0)
                    //{
                    //    node_name.append("_").append(std::to_string(i));
                    //}
                    //ng_split_op->set_name(node.name());
                    auto split_gnode = m_graph->add_node_and_edge(split_op, {input_gnode});
                    ret.push_back({node.name(), split_gnode});
                }
                return ret;
            }

            NamedNodeVector TranslateSplitVOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto length_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto split_dim_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int> lengths;
                bool status = GetValueFromNGraphOp<int>(length_gnode, &lengths);
                NNFUSION_CHECK(status);
                nnfusion::Shape shape = input_gnode->get_shape();
                int rank = shape.size();
                std::vector<size_t> lower(rank, 0);
                std::vector<size_t> upper(shape);

                std::vector<int64> split_dim_vec;
                status = GetValueFromNGraphOp<int64>(split_dim_gnode, &split_dim_vec);
                NNFUSION_CHECK(status);
                // there should be at least one element specified as axis and not more than
                // one as axis is 0-D
                NNFUSION_CHECK(split_dim_vec.size() == 1)
                    << "split_dim_tensor must have exactly one element.";
                status = CheckAxisDimInRange(split_dim_vec, rank);
                NNFUSION_CHECK(status);

                int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);

                // length: Length of size_splits
                int length = 0;
                int idx = -1;
                // Find out the total length of the splits and locate -1 's index, if any
                bool has_one_neg = false;
                for (int i = 0; i < lengths.size(); ++i)
                {
                    if (lengths[i] != -1)
                    {
                        length += lengths[i];
                    }
                    else
                    {
                        NNFUSION_CHECK(!has_one_neg) << "size_splits can only have one -1";

                        idx = i;
                        has_one_neg = true;
                    }
                }

                // Size splits must sum to the dimension of value along split_dim
                if (idx > 0)
                {
                    lengths[idx] = shape[split_dim] - length;
                }

                NNFUSION_CHECK((!has_one_neg && length == shape[split_dim]) ||
                               (has_one_neg && lengths[idx] >= 0))
                    << "The length of size_splits must sum to the value of the dimension along "
                       "split_dim";

                int cursor = 0;
                NamedNodeVector ret;

                if (lengths.size() != 1)
                {
                    for (int i = 0; i < lengths.size(); ++i)
                    {
                        lower[split_dim] = cursor;
                        cursor += lengths[i];
                        upper[split_dim] = cursor;
                        auto split_op = std::make_shared<op::Slice>(lower, upper);
                        //ng_split_op->set_name(node.name());

                        auto split_gnode = m_graph->add_node_and_edge(split_op, {input_gnode});
                        ret.push_back({node.name(), split_gnode});
                    }
                }
                else
                {
                    ret.push_back({node.name(), input_gnode});
                }

                return ret;
            }

            NamedNodeVector TranslateMeanOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto axes_gnode = GetInputNode(all_ng_nodes, node, 1);

                bool tf_keep_dims;
                if (GetNodeAttr(node.attr(), "keep_dims", tf_keep_dims) == false)
                {
                    if (GetNodeAttr(node.attr(), "keepdims", tf_keep_dims) == false)
                    {
                        tf_keep_dims = false;
                    }
                }

                nnfusion::Shape shape = input_gnode->get_shape();
                int rank = shape.size();

                std::vector<int64> mean_axes;
                bool status = GetValueFromNGraphOp<int64>(axes_gnode, &mean_axes);
                NNFUSION_CHECK(status);

                status = CheckAxisDimInRange(mean_axes, rank);
                NNFUSION_CHECK(status);

                std::vector<size_t> ng_reduction_axes_vect(mean_axes.size());
                std::transform(mean_axes.begin(),
                               mean_axes.end(),
                               ng_reduction_axes_vect.begin(),
                               [rank](int idx) { return idx + (idx < 0 ? (int)rank : 0); });
                nnfusion::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

                // todo: move to function ngraph::builder::mean?
                //std::shared_ptr<ngraph::Node> ng_mean =
                //    ngraph::builder::mean(input_gnode, ng_reduction_axes);

                auto xsum_op = std::make_shared<op::Sum>(ng_reduction_axes);
                auto xsum_gnode = m_graph->add_node_and_edge(xsum_op, {input_gnode});
                auto N = GetNumElements(input_gnode->get_shape(), ng_reduction_axes);
                const auto& et = xsum_gnode->get_element_type();
                auto xsum_shape = xsum_gnode->get_shape();
                std::vector<std::string> constant_values(nnfusion::shape_size(xsum_shape),
                                                         std::to_string(N));

                auto divisor_op = std::make_shared<op::Constant>(et, xsum_shape, constant_values);
                auto divisor_gnode = m_graph->add_node_and_edge(divisor_op, GNodeVector({}));

                auto mean_op = std::make_shared<op::Divide>();
                NamedNodeVector ret;
                // If keep_dims is specified we need to reshape to put back the reduced
                // axes, with length 1.
                if (tf_keep_dims)
                {
                    auto mean_gnode =
                        m_graph->add_node_and_edge(mean_op, {xsum_gnode, divisor_gnode});

                    nnfusion::Shape ng_result_shape_with_keep(rank);
                    for (size_t i = 0; i < rank; i++)
                    {
                        ng_result_shape_with_keep[i] =
                            ng_reduction_axes.count(i) == 0 ? shape[i] : 1;
                    }

                    nnfusion::AxisVector ng_axis_order(mean_gnode->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto reshape_mean_op =
                        std::make_shared<op::Reshape>(ng_axis_order, ng_result_shape_with_keep);
                    reshape_mean_op->set_name(node.name());
                    auto reshape_mean_gnode =
                        m_graph->add_node_and_edge(reshape_mean_op, {mean_gnode});
                    ret.push_back({node.name(), reshape_mean_gnode});
                }
                else
                {
                    mean_op->set_name(node.name());
                    auto mean_gnode =
                        m_graph->add_node_and_edge(mean_op, {xsum_gnode, divisor_gnode});
                    ret.push_back({node.name(), mean_gnode});
                }

                return ret;
            }

            NamedNodeVector TranslateSliceOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto begin_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto size_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> lower_vec;
                std::vector<int64> size_vec;
                bool status = GetValueFromNGraphOp<int64>(begin_gnode, &lower_vec);
                NNFUSION_CHECK(status);
                status = GetValueFromNGraphOp<int64>(size_gnode, &size_vec);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(lower_vec.size() == size_vec.size())
                    << "Cannot translate sliceop: Size of lower = " << lower_vec.size()
                    << ", size of size_vec = " << size_vec.size() << ". Expected them to match.";

                std::vector<int> upper_vec(lower_vec.size());
                const auto input_shape = input_gnode->get_shape();
                std::stringstream err_stream;
                std::string err_msg;
                for (size_t i = 0; i < size_vec.size(); i++)
                {
                    if (size_vec[i] != -1)
                    {
                        upper_vec[i] = lower_vec[i] + size_vec[i];
                    }
                    else
                    {
                        // support -1 for size_vec, to the end of the tensor
                        upper_vec[i] = input_shape[i];
                    }

                    // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
                    if (0 > lower_vec[i])
                    {
                        err_stream << "lower < 0: " << lower_vec[i]
                                   << ". It should have been positive.\n";
                    }
                    if (lower_vec[i] > upper_vec[i])
                    {
                        err_stream << "upper < lower: upper = " << upper_vec[i]
                                   << ", lower = " << lower_vec[i] << "\n";
                    }
                    if (upper_vec[i] > input_shape[i])
                    {
                        err_stream << "dim < upper: dim = " << input_shape[i]
                                   << ", upper = " << upper_vec[i] << "\n";
                    }

                    err_msg = err_stream.str();
                    NNFUSION_CHECK(err_msg.empty()) << "Cannot translate sliceop at position " << i
                                                    << " of " << size_vec.size()
                                                    << ". The reasons are:\n"
                                                    << err_msg;
                }

                std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
                std::vector<size_t> u(upper_vec.begin(), upper_vec.end());
                auto slice_op = std::make_shared<op::Slice>(l, u);
                slice_op->set_name(node.name());
                auto slice_gnode = m_graph->add_node_and_edge(slice_op, {input_gnode});

                NamedNodeVector ret{{node.name(), slice_gnode}};
                return ret;
            }

            NamedNodeVector TranslateTransposeOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto permutation_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> permutation;
                bool status = GetValueFromNGraphOp<int64>(permutation_gnode, &permutation);
                NNFUSION_CHECK(status);

                nnfusion::AxisVector ng_axis_order;
                ng_axis_order.reserve(permutation.size());

                for (auto i : permutation)
                {
                    ng_axis_order.push_back(i);
                }

                nnfusion::op::OpConfig::any myConfig;
                myConfig["axes_order"] = ng_axis_order;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnode});
                NamedNodeVector ret{{node.name(), input_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateTransposeToReshapeOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto permutation_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> permutation;
                bool status = GetValueFromNGraphOp<int64>(permutation_gnode, &permutation);
                NNFUSION_CHECK(status);

                // Check to make sure that the permutation requested for transpose
                // is valid for example:
                // - it should not have duplicates,
                // - it should have all the dimensions.

                auto input_rank = input_gnode->get_shape().size();
                vector<bool> count(input_rank, false);
                for (auto p : permutation)
                {
                    if (0 <= p && p < input_rank)
                    {
                        count[p] = true;
                    }
                }
                for (int i = 0; i < input_rank; i++)
                {
                    NNFUSION_CHECK(count[i]) << i << " is missing from {" << join(permutation)
                                             << "}.";
                }

                nnfusion::AxisVector ng_axis_order;
                ng_axis_order.reserve(permutation.size());

                for (auto i : permutation)
                {
                    ng_axis_order.push_back(i);
                }

                auto trans_gnode = graph::numpy_transpose(input_gnode, ng_axis_order);
                m_graph->add_node(trans_gnode);
                m_graph->add_edge(input_gnode, 0, trans_gnode, 0);

                trans_gnode->get_op_ptr()->set_name(node.name());
                trans_gnode->set_name(node.name());
                NamedNodeVector ret{{node.name(), trans_gnode}};
                return ret;
            }

            NamedNodeVector TranslateOneHotOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto features_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto depth_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto ong_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto off_gnode = GetInputNode(all_ng_nodes, node, 3);

                auto features_shape = features_gnode->get_shape();
                auto features_rank = features_shape.size();

                std::vector<int> depth;
                bool status = GetValueFromNGraphOp<int>(depth_gnode, &depth);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(depth.size() == 1)
                    << "OneHot Op: depth of one hot dimension must be scalar " << depth.size();

                std::vector<float> on_value;
                status = GetValueFromNGraphOp<float>(ong_gnode, &on_value);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(on_value.size() == 1)
                    << "OneHot Op: on value of one hot dimension must be scalar "
                    << on_value.size();

                std::vector<float> off_value;
                status = GetValueFromNGraphOp<float>(off_gnode, &off_value);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(off_value.size() == 1)
                    << "OneHot Op: off value of one hot dimension must be scalar "
                    << off_value.size();

                int one_hot_axis;
                status = GetNodeAttr(node.attr(), "axis", one_hot_axis);
                NNFUSION_CHECK(status);
                tensorflow::DataType dtype;
                status = GetNodeAttr(node.attr(), "T", dtype);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et;
                status = TFDataTypeToNNFusionElementType(dtype, &nnfusion_et);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(nnfusion_et == nnfusion::element::f32);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["axis"] = one_hot_axis;
                myConfig["depth"] = depth[0];
                myConfig["off_value"] = off_value[0];
                myConfig["on_value"] = on_value[0];
                myConfig["T"] = nnfusion_et.c_type_string();

                //features_gnode->set_output_type(0, nnfusion_et, features_gnode->get_shape());

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {features_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateStopGradientOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);

                nnfusion::op::OpConfig::any myConfig;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateGatherV2Op(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto input_coords_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto axis_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int64> tf_axis;
                bool status = GetValueFromNGraphOp<int64>(axis_gnode, &tf_axis);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(tf_axis.size() == 1) << "Found axis in GatherV2 op (" << node.name()
                                                    << ") translation to be non scalar, of size "
                                                    << tf_axis.size();

                nnfusion::op::OpConfig::any myConfig;
                myConfig["axis"] = tf_axis[0];

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {input_gnode, input_coords_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateAddNOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // Use this to get all the inputs of current node.
                // Use GetInputNode(..., ..., id) to get the input identified by
                // id.
                auto input_gnodes = GetAllInputNode(all_ng_nodes, node);

                nnfusion::op::OpConfig::any myConfig;

                // Since Ngraph doesn't have AddN, so we use GenericOp to
                // represent the AddN.
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    node.name(), // Node name, looks like "tf_model/add_n";
                    node.op(),   // Operator name, looks like "AddN";
                    myConfig);   // The configuration we generated above;

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnodes});
                // Return the node vecoter, this is one tf-node to one nnfusion-node case,
                // if your code converts one tf-node into several nnfusion-nodes, you can
                // refer BiasAdd, which is converted to Broadcast and Add;
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslatePackOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                const int input_cnt = node.input_size();
                NNFUSION_CHECK(input_cnt >= 1) << "\"" << node.name()
                                               << "\" requires at least 1 inputs, got " << input_cnt
                                               << " instead";

                int pack_axis = 0;
                bool status = GetNodeAttr(node.attr(), "axis", pack_axis);
                NNFUSION_CHECK(status);

                GNodeVector input_gnodes;
                for (int i = 0; i < input_cnt; i++)
                {
                    auto ng_arg = GetInputNode(all_ng_nodes, node, i);
                    input_gnodes.push_back(ng_arg);
                }

                if (pack_axis < 0)
                {
                    pack_axis += int64(input_gnodes[0]->get_shape().size() + 1);
                }

                if (true)
                {
                    // option1, covert pack to combination of expand_dim and concat
                    auto& input_shape = input_gnodes[0]->get_shape();
                    auto input_shape_size = input_shape.size();

                    // expand_dim/reshape
                    auto new_dim_shape = input_shape;
                    new_dim_shape.insert(new_dim_shape.begin() + size_t(pack_axis), 1);
                    std::vector<size_t> shape_dimensions(input_shape_size);
                    std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);

                    GNodeVector reshaped_input_gnodes;

                    for (int i = 0; i < input_gnodes.size(); i++)
                    {
                        auto input_gnode = input_gnodes[i];
                        auto reshape_input_op =
                            std::make_shared<op::Reshape>(shape_dimensions, new_dim_shape);
                        reshape_input_op->set_name(node.name() + "_reshape_" + std::to_string(i));
                        auto reshape_input_gnode =
                            m_graph->add_node_and_edge(reshape_input_op, {input_gnode});
                        reshaped_input_gnodes.push_back(reshape_input_gnode);
                    }

                    // concat
                    auto concat_op = std::make_shared<op::Concat>(size_t(pack_axis));

                    concat_op->set_name(node.name());
                    auto concat_gnode =
                        m_graph->add_node_and_edge(concat_op, reshaped_input_gnodes);
                    NamedNodeVector ret{{node.name(), concat_gnode}};
                    return ret;
                }
                else
                {
                    // TODO: option2, implement pack kernel
                    NNFUSION_CHECK_FAIL() << "Pack kernel not implemented yet";

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = pack_axis;

                    auto generic_op =
                        std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_gnodes);
                    NamedNodeVector ret{{node.name(), generic_gnode}};
                    return ret;
                }
            }

            NamedNodeVector TranslateAllOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto axis_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int> tf_axis;
                bool status = GetValueFromNGraphOp<int>(axis_gnode, &tf_axis);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(tf_axis.size() <= 1) << "Found axis in All op (" << node.name()
                                                    << ") translation to be non scalar, of size "
                                                    << tf_axis.size();

                bool keep_dims = false;
                status = GetNodeAttr(node.attr(), "keep_dims", keep_dims);
                NNFUSION_CHECK(status);

                nnfusion::op::OpConfig::any myConfig;
                if (tf_axis.size() > 0)
                {
                    myConfig["axis"] = tf_axis[0];
                }
                myConfig["keep_dims"] = keep_dims;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateAssignOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto ref_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto val_gnode = GetInputNode(all_ng_nodes, node, 1);

                nnfusion::op::OpConfig::any myConfig;
                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {ref_gnode, val_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateAssignSubOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto ref_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto val_gnode = GetInputNode(all_ng_nodes, node, 1);

                nnfusion::op::OpConfig::any myConfig;
                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {ref_gnode, val_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateApplyAdamOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto var_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto m_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto v_gnode = GetInputNode(all_ng_nodes, node, 2);
                // { const
                auto beta1_pow_gnode = GetInputNode(all_ng_nodes, node, 3);
                auto beta2_pow_gnode = GetInputNode(all_ng_nodes, node, 4);
                // }
                auto lr_gnode = GetInputNode(all_ng_nodes, node, 5);
                // { const
                auto beta1_gnode = GetInputNode(all_ng_nodes, node, 6);
                auto beta2_gnode = GetInputNode(all_ng_nodes, node, 7);
                auto epsilon_gnode = GetInputNode(all_ng_nodes, node, 8);
                // }
                auto grad_gnode = GetInputNode(all_ng_nodes, node, 9);

                nnfusion::op::OpConfig::any myConfig;
                vector<float> t;
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(epsilon_gnode, &t));
                myConfig["epsilon"] = t[0];
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(beta1_gnode, &t));
                myConfig["beta1"] = t[0];
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(beta2_gnode, &t));
                myConfig["beta2"] = t[0];
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(beta1_pow_gnode, &t));
                myConfig["beta1_pow"] = t[0];
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(beta1_pow_gnode, &t));
                myConfig["beta2_pow"] = t[0];

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode = m_graph->add_node_and_edge(
                    generic_op, {var_gnode, m_gnode, v_gnode, lr_gnode, grad_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateApplyMomentumOp(const tensorflow::NodeDef& node,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto var_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto accum_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto lr_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto grad_gnode = GetInputNode(all_ng_nodes, node, 3);
                auto momentum_gnode = GetInputNode(all_ng_nodes, node, 4);

                std::vector<float> lr_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(lr_gnode, &lr_value))
                    << "We only accept the lr as Constant.";
                NNFUSION_CHECK(lr_value.size() == 1);

                std::vector<float> momentum_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(momentum_gnode, &momentum_value))
                    << "We only accept the momentum as Constant.";
                NNFUSION_CHECK(momentum_value.size() == 1);

                bool use_nesterov = false;
                bool status = GetNodeAttr(node.attr(), "use_nesterov", use_nesterov);
                NNFUSION_CHECK(status);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["use_nesterov"] = use_nesterov;
                myConfig["lr"] = lr_value[0];
                myConfig["momentum"] = momentum_value[0];

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {var_gnode, accum_gnode, grad_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateSparseApplyMomentumOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto var_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto accum_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto lr_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto grad_gnode = GetInputNode(all_ng_nodes, node, 3);
                auto indices_gnode = GetInputNode(all_ng_nodes, node, 4);
                auto momentum_gnode = GetInputNode(all_ng_nodes, node, 5);

                auto& indices_shape = indices_gnode->get_shape();
                NNFUSION_CHECK(nnfusion::is_vector(indices_shape))
                    << "indices must be one-dimensional";
                auto& var_shape = var_gnode->get_shape();
                NNFUSION_CHECK(nnfusion::is_vector_or_higher(var_shape))
                    << "var must be at least 1 dimensional";

                std::vector<int64> indices;
                bool status = GetValueFromNGraphOp<int64>(indices_gnode, &indices);
                NNFUSION_CHECK(status);

                std::vector<float> lr_value;
                status = GetValueFromNGraphOp<float>(lr_gnode, &lr_value);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(lr_value.size() == 1);

                std::vector<float> momentum_value;
                status = GetValueFromNGraphOp<float>(momentum_gnode, &momentum_value);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(momentum_value.size() == 1);

                bool use_nesterov = false;
                status = GetNodeAttr(node.attr(), "use_nesterov", use_nesterov);
                NNFUSION_CHECK(status);

                tensorflow::DataType dtype;
                status = GetNodeAttr(node.attr(), "Tindices", dtype);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et;
                status = TFDataTypeToNNFusionElementType(dtype, &nnfusion_et);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(nnfusion_et == nnfusion::element::i32 ||
                               nnfusion_et == nnfusion::element::i64);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["use_nesterov"] = use_nesterov;
                myConfig["lr"] = lr_value[0];
                myConfig["momentum"] = momentum_value[0];
                myConfig["indices"] = indices;
                myConfig["Tindices"] = nnfusion_et.c_type_string();

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {var_gnode, accum_gnode, grad_gnode});

                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateSqueezeOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                size_t input_dims = input_gnode->get_shape().size();

                std::vector<int32> tf_axis;
                bool status = GetNodeAttr(node.attr(), "squeeze_dims", tf_axis);
                NNFUSION_CHECK(status);

                // If input dimension is negative, make it positive
                for (int i = 0; i < tf_axis.size(); i++)
                {
                    tf_axis[i] = tf_axis[i] < 0 ? (int32)(input_dims) + tf_axis[i] : tf_axis[i];
                }

                std::set<int> axis_set(tf_axis.begin(), tf_axis.end());
                nnfusion::Shape input_shape = input_gnode->get_shape();
                std::vector<int> dims;

                if (axis_set.size() == 0)
                {
                    for (size_t i = 0; i < input_dims; i++)
                    {
                        if (input_shape[i] > 1)
                        {
                            dims.push_back(input_shape[i]);
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < input_dims; i++)
                    {
                        bool skip = false;
                        if (axis_set.find(i) != axis_set.end())
                        {
                            NNFUSION_CHECK(input_shape[i] == 1)
                                << "Tried to explicitly squeeze dimension " << i
                                << " but dimension was not 1: " << input_shape[i];
                            skip = true;
                        }
                        if (!skip)
                        {
                            dims.push_back(input_shape[i]);
                        }
                    }
                }

                nnfusion::Shape output_shape(dims.size());
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    output_shape[i] = dims[i];
                }

                nnfusion::AxisVector ng_axis_order(input_gnode->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                auto reshape_op = std::make_shared<op::Reshape>(ng_axis_order, output_shape);
                reshape_op->set_name(node.name());
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_gnode});
                NamedNodeVector ret{{node.name(), reshape_gnode}};
                return ret;
            }

            NamedNodeVector TranslateExpandDimsOp(const tensorflow::NodeDef& node,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto dim_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> dim_vec;
                bool status = GetValueFromNGraphOp<int64>(dim_gnode, &dim_vec);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(dim_vec.size() == 1)
                    << "The size of argument dim is not 1 for ExpandDims";

                auto& shape = input_gnode->get_shape();
                auto shape_size = shape.size();
                if (dim_vec[0] < 0)
                {
                    // allow range [-rank(input) - 1, rank(input)]
                    // where -1 append new axis at the end
                    dim_vec[0] = shape_size + dim_vec[0] + 1;
                }

                auto out_shape = shape;
                out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);
                std::vector<size_t> shape_dimensions(shape.size());
                std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);

                auto reshape_op = std::make_shared<op::Reshape>(shape_dimensions, out_shape);
                reshape_op->set_name(node.name());

                auto reshape_gnode =
                    m_graph->add_node_and_edge(reshape_op, GNodeVector({input_gnode}));
                NamedNodeVector ret{{node.name(), reshape_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateSquaredDifferenceOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::tie(lhs_gnode, rhs_gnode) =
                    graph::numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                auto diff_op = std::make_shared<op::Subtract>();
                auto diff_gnode = m_graph->add_node_and_edge(diff_op, {lhs_gnode, rhs_gnode});

                auto multiply_op = std::make_shared<op::Multiply>();
                multiply_op->set_name(node.name());
                auto multiply_gnode =
                    m_graph->add_node_and_edge(multiply_op, {diff_gnode, diff_gnode});
                NamedNodeVector ret{{node.name(), multiply_gnode}};
                return ret;
            }

            NamedNodeVector TranslateRangeOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes;
                auto starg_gnode = GetInputNode(all_ng_nodes, node, 0);
                input_gnodes.push_back(starg_gnode);
                auto limit_gnode = GetInputNode(all_ng_nodes, node, 1);
                input_gnodes.push_back(limit_gnode);
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 2);
                input_gnodes.push_back(delta_gnode);

                std::vector<int64> start_vec;
                NNFUSION_CHECK(GetValueFromNGraphOp<int64>(starg_gnode, &start_vec) == true);
                NNFUSION_CHECK(start_vec.size() > 0);
                std::vector<int64> limit_vec;
                NNFUSION_CHECK(GetValueFromNGraphOp<int64>(limit_gnode, &limit_vec) == true);
                NNFUSION_CHECK(limit_vec.size() > 0);
                std::vector<int64> delta_vec;
                NNFUSION_CHECK(GetValueFromNGraphOp<int64>(delta_gnode, &delta_vec) == true);
                NNFUSION_CHECK(delta_vec.size() > 0);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["start"] = start_vec[0];
                myConfig["limit"] = limit_vec[0];
                myConfig["delta"] = delta_vec[0];

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_gnodes);
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateReduceAnyOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                NamedNodeVector ret;
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto axes_gnode = GetInputNode(all_ng_nodes, node, 1);
                NNFUSION_CHECK(input_gnode->get_output_element_type(0).c_type_string() == "char")
                    << "Input tensor of ReduceAny op should be bool(underlying char), but given as "
                    << input_gnode->get_output_element_type(0).c_type_string() << ".";

                nnfusion::Shape output_shape;
                vector<int32_t> axis;
                bool status = GetValueFromNGraphOp<int32_t>(axes_gnode, &axis);
                auto input_rank = input_gnode->get_shape().size();

                std::vector<size_t> ng_reduction_axes_vect(axis.size());
                std::transform(
                    axis.begin(),
                    axis.end(),
                    ng_reduction_axes_vect.begin(),
                    [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });

                nnfusion::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

                auto or_op = std::make_shared<op::ReduceAny>(ng_reduction_axes);
                or_op->set_name(node.name());
                auto ret_gnode = m_graph->add_node_and_edge(or_op, {input_gnode});

                bool keepdims;
                if (!GetNodeAttr(node.attr(), "keepdims", keepdims))
                    NNFUSION_CHECK(GetNodeAttr(node.attr(), "keep_dims", keepdims));

                if (keepdims)
                {
                    nnfusion::AxisVector ng_axis_order(ret_gnode->get_output_shape(0).size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                    auto reshape_op = std::make_shared<op::Reshape>(
                        ng_axis_order, input_gnode->get_output_shape(0));
                    reshape_op->set_name(node.name() + "_keepdims");
                    ret_gnode = m_graph->add_node_and_edge(reshape_op, {ret_gnode});
                }

                ret.push_back({node.name(), ret_gnode});
                return ret;
            }

            NamedNodeVector TranslateRsqrtGradOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 1);

                //`grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
                // Create a constant tensor populated with the value 3.
                auto et = input_gnode->get_element_type();
                auto shape = input_gnode->get_shape();
                std::vector<std::string> constant_values(nnfusion::shape_size(shape), "3");

                auto exponent_op = std::make_shared<op::Constant>(et, shape, constant_values);
                auto exponent_gnode = m_graph->add_node_and_edge(exponent_op, GNodeVector({}));
                // Raise each element of the input to the power 3.
                auto pow_op = std::make_shared<op::Power>();
                auto pow_gnode = m_graph->add_node_and_edge(pow_op, {input_gnode, exponent_gnode});

                // Create a constant tensor populated with the value -1/2.
                std::vector<std::string> constant_diff(nnfusion::shape_size(shape), "-0.5");
                auto diff_op = std::make_shared<op::Constant>(et, shape, constant_diff);
                auto diff_gnode = m_graph->add_node_and_edge(diff_op, GNodeVector({}));

                auto multiply_op = std::make_shared<op::Multiply>();
                auto multiply_gnode =
                    m_graph->add_node_and_edge(multiply_op, {pow_gnode, delta_gnode});
                auto ret_op = std::make_shared<op::Multiply>();
                ret_op->set_name(node.name());
                auto ret_gnode = m_graph->add_node_and_edge(ret_op, {multiply_gnode, diff_gnode});
                NamedNodeVector ret{{node.name(), ret_gnode}};
                return ret;
            }

            NamedNodeVector TranslateStridedSliceOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // TODO: implement new_axis_mask, ellipsis_mask
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto begin_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto end_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto stride_gnode = GetInputNode(all_ng_nodes, node, 3);

                std::vector<int64> begin_vec;
                bool status = GetValueFromNGraphOp<int64>(begin_gnode, &begin_vec);
                NNFUSION_CHECK(status);
                std::vector<int64> end_vec;
                status = GetValueFromNGraphOp<int64>(end_gnode, &end_vec);
                NNFUSION_CHECK(status);
                std::vector<int64> stride_vec;
                status = GetValueFromNGraphOp<int64>(stride_gnode, &stride_vec);
                NNFUSION_CHECK(status);

                int tf_shrink_axis_mask;
                status = GetNodeAttr(node.attr(), "shrink_axis_mask", tf_shrink_axis_mask);
                NNFUSION_CHECK(status);

                int tf_end_mask;
                status = GetNodeAttr(node.attr(), "end_mask", tf_end_mask);
                NNFUSION_CHECK(status);

                int tf_begin_mask;
                status = GetNodeAttr(node.attr(), "begin_mask", tf_begin_mask);
                NNFUSION_CHECK(status);

                int tf_new_axis_mask;
                status = GetNodeAttr(node.attr(), "new_axis_mask", tf_new_axis_mask);
                NNFUSION_CHECK(status);

                int tf_ellipsis_mask;
                status = GetNodeAttr(node.attr(), "ellipsis_mask", tf_ellipsis_mask);
                NNFUSION_CHECK(status);

                auto& input_shape = input_gnode->get_shape();

                // Summary: Convert tf indexes (-inf, inf) to clamped_begin_idx [0, d] and
                // clamped_end_idx [-1, d], which are then converted to ngraph indexes [0, d]
                // tf->ng is done through tf_to_ng, which calls clamper, which converts
                // tf->clamped

                // Graph/function for tf->cmapled
                //           |    .......     <-- y = max_val (max_val = 5)
                //          .|   .
                //         . |  .
                //        .  | .              <-- y = x>=0 ? x : x+max_val
                //       .   |.
                // -.-.-.----.------------    <-- y = 0 (for inclusive)
                //  * *      |                <-- y = -1 (for exclusive)
                //           |
                // X axis: TF indexes. Y axis: Clamped indexes

                // clamper is a function that implements the graph above.
                // For inclusive, the graph is clamped at 0 and dim-1
                // Given dimension d, [0, d-1] are valid locations.
                // -1 represents std::rend(). d represents std::end().
                // These two are useful for representing exclusive boundaries for end-ranges
                // Example for dim = 3:
                // ranges:                 (-inf,-d)|   [-d,0)    |[0,d-1]|(d-1,inf)
                // TF index:                  -5 -4 |-3  -2 -1    | 0 1 2 | 3 4 5
                // clamped begin (inclusive):  0  0 | 0   1  2    | 0 1 2 | 3 3 3
                // clamped end (exclusive):   -1 -1 | 0   1  2    | 0 1 2 | 3 3 3
                auto clamper = [](int idx, size_t dim, bool inclusive) {
                    // if idx is in [-(d-1), d-1], then its same for both inclusive and
                    // exclusive
                    // The first 2 cases breaks down this range
                    if (idx >= 0 && idx <= (static_cast<int>(dim) - 1))
                    {
                        return idx;
                    }
                    else if (idx < 0 && idx + static_cast<int>(dim) >= 0)
                    { // careful not to do idx >= -dim
                        // (since dim is unsigned)
                        return idx + static_cast<int>(
                                         dim); // Type casting to int to enable unambiguous auto
                                               // type inference of return type
                    }
                    else if (idx > static_cast<int>(dim) - 1)
                    {
                        return static_cast<int>(dim);
                    }
                    else if (idx + static_cast<int>(dim) < 0)
                    {
                        // The next case handles the clamping (differently for inclusive and
                        // exclusive cases)

                        // careful not to do idx < -dim (since dim is unsigned)
                        return 0 - (inclusive ? 0 : 1);
                    }
                    // Default case
                    return 0;
                };

                auto tf_to_ng = [clamper](int tf_begin_idx,
                                          int tf_end_idx,
                                          int tf_stride,
                                          size_t dim,
                                          bool begin_mask,
                                          bool end_mask,
                                          bool shrink_mask) {
                    // if begin mask is present, depending on stride sign use 0 (std::begin) or
                    // dim-1 (std::rbegin)
                    // clamped_end_idx could line in [-1, d]
                    int tf_ignore_begin_if_needed =
                        begin_mask ? (tf_stride > 0 ? 0 : dim - 1) : tf_begin_idx;
                    // if end mask is present, depending on stride sign use -1 (std::rend) or
                    // dim (std::end).
                    // However note, we cannot set to -1, since it has another meaning, hence
                    // setting to -(dim+1), which would translate to -1 in clamped coordinates
                    // take care to convert dim from sixze_t to int
                    int tf_ignore_end_if_needed =
                        end_mask ? (tf_stride > 0 ? dim : (-((int)dim + 1))) : tf_end_idx;
                    // using size_t for clamped_begin_idx because: clamped_begin_idx is
                    // inclusive, so it must lie in [0, dim-1]
                    size_t clamped_begin_idx = clamper(tf_ignore_begin_if_needed, dim, true);
                    int64 clamped_end_idx = clamper(
                        shrink_mask ? clamped_begin_idx + 1 : tf_ignore_end_if_needed, dim, false);

                    // Now we have converted semantically non-monotonic and unbounded TF indexes
                    // (-inf, inf) to bounded and monotonic clamped indexes [-1, d]
                    // Now we need to convert clamped indexes [-1, d] to ngraph indexes [0, d]
                    // (taking care of reversal in case of negative strides)

                    size_t needs_reverse = 0;
                    size_t ng_begin_idx, ng_end_idx;

                    if (!shrink_mask)
                    {
                        if (clamped_begin_idx == clamped_end_idx)
                        {
                            // Empty due to matching indexes
                            ng_begin_idx = clamped_begin_idx;
                            // Type safety: clamped_begin_idx == clamped_end_idx implies,
                            // clamped_end_idx!=-1 (since clamped_begin_idx cannot be -1), hence end
                            // index assignment is type safe
                            ng_end_idx = clamped_end_idx;
                        }
                        else
                        { // In the whole of this else: clamped_begin_idx !=
                            // clamped_end_idx, so !(a < b) iff a > b and vice versa when
                            // comparing the indexes
                            // take care to use (int) typecase when comparing int and size_t
                            if (((int)clamped_begin_idx < clamped_end_idx) != (tf_stride > 0))
                            {
                                // Empty due to mismatching directions
                                ng_begin_idx = clamped_begin_idx;
                                // Type safe: since clamped_begin_idx is size_t (>0)
                                // [0:-4:1] in TF would convert to [0:-1:1] in clamped domain. hence
                                // we do not assign ng_end_idx = clamped_end_idx (which would not be
                                // type safe due to the -1)
                                ng_end_idx = clamped_begin_idx;
                                // Any assignment where ng_begin_idx = ng_end_idx = x (where 0 <= x <=
                                // d-1) would have worked for the 2 empty cases above
                            }
                            // Anything after this is non-empty. Anything before this has dealt with
                            // empty cases
                            else
                            {
                                // in this case either (clamped_begin_idx < clamped_end_idx &&
                                // tf_stride > 0) or (clamped_begin_idx > clamped_end_idx && tf_stride
                                // < 0)
                                // that is clamped_begin_idx < clamped_end_idx <==> tf_stride > 0.
                                // hence using only 1 of the clauses is enough
                                if (tf_stride > 0)
                                {
                                    ng_begin_idx = clamped_begin_idx;
                                    // Type safety: tf_stride > 0 ==> clamped_begin_idx <
                                    // clamped_end_idx. clamped_begin_idx could be 0,
                                    // which means clamped_end_idx > 0. Hence type-safe
                                    ng_end_idx = clamped_end_idx;
                                }
                                else
                                { // clamped_begin_idx > clamped_end_idx, tf_stride < 0

                                    // clamped_begin_idx is [0, d] && clamped_begin_idx >
                                    // clamped_end_idx,
                                    // which implies clamped_end_idx is [-1,d-1]
                                    // Type safety: With clamped_end_idx in [-1,d-1],
                                    // dim - 1 - clamped_end_idx is in [0, dim]. Hence type safe
                                    ng_end_idx = dim - 1 - clamped_end_idx;

                                    if (clamped_begin_idx == dim)
                                    {
                                        clamped_begin_idx = dim - 1;
                                    }
                                    // Note clamped_begin_idx != dim here.
                                    // If clamped_begin_idx==dim && clamped_end_idx==dim, then "Empty
                                    // due to matching indexes" handles it
                                    // If clamped_begin_idx==dim && clamped_end_idx<dim, then 2 cases:
                                    //   tf_stride > 0: then "Empty due to mismatching directions"
                                    //   handles it
                                    //   tf_stride < 0: Then we set it to dim-1 above
                                    // Consider the case of dim=3, where in tf notation we have:
                                    // [4:1:-1], in clampe notation, we get [3:1:-1], which really means
                                    // [2:1:-1]

                                    // Type safety: Since clamped_begin_idx is [0, d-1] here, it is type
                                    // safe
                                    ng_begin_idx = dim - 1 - clamped_begin_idx;
                                    needs_reverse = 1;
                                }
                            }
                        }
                    }
                    else
                    {
                        // cases when clamped indexes are in [0,d] and hence can be directly
                        // copied
                        // TODO: what about tf_begin=d, shrink=T, then clamped_end_idx = d, so a
                        // 0-d axis.
                        // But since shrink is on, that is reshaped and the 0-d axis is removed?
                        // Is that a valid config, as shrink_axis must get an axis with dim = 1,
                        // right?

                        ng_begin_idx = clamped_begin_idx;
                        ng_end_idx = clamped_end_idx;
                    }
                    return std::make_tuple(
                        ng_begin_idx, ng_end_idx, std::abs(tf_stride), needs_reverse);
                };

                auto extract_bit = [](int bit_mask, int bit_location) {
                    return (bit_mask & (1 << bit_location)) != 0;
                };

                auto dim_vec = input_gnode->get_shape();
                auto in_rank = dim_vec.size();

                /*
                NNFUSION_CHECK(begin_vec.size() <= in_rank)
                    << "Index out of range using input dim " << begin_vec.size()
                    << "; input has only " << in_rank << " dims";
                */

                // TODO/Note/Question: Are begin, end and stride vectors are of equal length

                // begin, end and stride vectors may not have same size as input rank, hence
                // initialize them with 0, dim and 1 respectively
                vector<size_t> ng_begin_vec(in_rank, 0), ng_stride_vec(in_rank, 1);
                vector<size_t> ng_end_vec(dim_vec);
                vector<size_t> ng_needs_reversal(in_rank, 0); // should have been a
                                                              // vector<bool>, but it is
                                                              // optimized, so tie won't
                                                              // work. Hence using size_t
                auto min_rank = std::min(in_rank, begin_vec.size());
                for (int dim_idx = 0; dim_idx < min_rank; dim_idx++)
                {
                    std::tie(ng_begin_vec[dim_idx],
                             ng_end_vec[dim_idx],
                             ng_stride_vec[dim_idx],
                             ng_needs_reversal[dim_idx]) =
                        tf_to_ng(begin_vec[dim_idx],
                                 end_vec[dim_idx],
                                 stride_vec[dim_idx],
                                 dim_vec[dim_idx],
                                 extract_bit(tf_begin_mask, dim_idx),
                                 extract_bit(tf_end_mask, dim_idx),
                                 extract_bit(tf_shrink_axis_mask, dim_idx));
                }

                // filter out negative stride dimensions
                vector<size_t> neg_strides;
                for (int dim_idx = 0; dim_idx < min_rank; dim_idx++)
                {
                    if (ng_needs_reversal[dim_idx])
                    {
                        neg_strides.push_back(dim_idx);
                    }
                }
                std::cout << join(ng_stride_vec) << std::endl;

                // atleast one stride was negative, in which case reverse the input
                if (neg_strides.size() > 0)
                {
                    auto reverse_input_op = std::make_shared<op::Reverse>(neg_strides);
                    input_gnode = m_graph->add_node_and_edge(reverse_input_op, {input_gnode});
                }

                std::cout << join(ng_stride_vec) << std::endl;

                auto strided_slice_op =
                    std::make_shared<op::Slice>(ng_begin_vec, ng_end_vec, ng_stride_vec);
                auto strided_slice_gnode =
                    m_graph->add_node_and_edge(strided_slice_op, {input_gnode});

                if (tf_shrink_axis_mask)
                {
                    int64 shrink_axis_mask = tf_shrink_axis_mask;
                    vector<size_t> output_shape;

                    // Note: do not use rank instead of ng_begin_vec.size()
                    // since ng_begin_vec.size() can be less than rank, and
                    // shrink_mask will have atmost ng_begin_vec.size() elements
                    for (int i = 0; i < ng_begin_vec.size(); i++)
                    {
                        if ((shrink_axis_mask & 1) != 1)
                        {
                            output_shape.push_back(ng_end_vec[i] - ng_begin_vec[i]);
                        }
                        else
                        {
                            // TODO: must it equal 1 or can it be 0 too?
                            NNFUSION_CHECK(ng_end_vec[i] - ng_begin_vec[i] <= 1)
                                << "Trying to shrink specification " << i
                                << "where tf begin, end, strides are: " << begin_vec[i] << ":"
                                << end_vec[i] << ":" << stride_vec[i]
                                << ". nGraph begin, end, stride are: " << ng_begin_vec[i] << ":"
                                << ng_end_vec[i] << ":" << ng_stride_vec[i]
                                << ". nGraph's begin and end have difference greater than "
                                   "1";
                        }
                        shrink_axis_mask >>= 1;
                    }

                    nnfusion::Shape ng_final_shape(output_shape);
                    nnfusion::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                    auto reshape_strided_slice_op =
                        std::make_shared<op::Reshape>(ng_axis_order, ng_final_shape);
                    strided_slice_gnode =
                        m_graph->add_node_and_edge(reshape_strided_slice_op, {strided_slice_gnode});
                }

                // TODO: assert size in this dim was 1
                // TODO: assert new_axis_mask and tf_shrink_axis_mask are not set at the same
                // time?
                // TODO: tf_new_axis_mask can exceed rank
                // Raise each element of the input to the power -0.5.

                strided_slice_gnode->set_name(node.name());
                strided_slice_gnode->get_op_ptr()->set_name(node.name());
                NamedNodeVector ret{{node.name(), strided_slice_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateStridedSliceGradOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto x_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto begin_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto end_gnode = GetInputNode(all_ng_nodes, node, 2);
                auto strides_gnode = GetInputNode(all_ng_nodes, node, 3);
                auto grad_gnode = GetInputNode(all_ng_nodes, node, 4);

                std::vector<int32> x_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<int32>(x_gnode, &x_value))
                    << "StridedSliceGradOp currently do not support dynamic output tensor shape";
                auto x_shape = x_gnode->get_shape();
                auto x_const_op = std::make_shared<op::Constant>(element::i32, x_shape, x_value);
                auto x_const_gnode = m_graph->add_node_and_edge(x_const_op, GNodeVector({}));

                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    node.name(),                    // Node name, looks like "tf_model/add_n";
                    node.op(),                      // Operator name, looks like "AddN";
                    nnfusion::op::OpConfig::any{}); // The configuration we generated above;

                int32_t dims = x_shape[0];
                // Check and replace Constant
                if (begin_gnode->get_output_shape(0)[0] < dims)
                {
                    vector<int32_t> vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int32_t>(begin_gnode, &vec));
                    for (size_t i = vec.size(); i < dims; i++)
                        vec.push_back(0);
                    std::cout << dims << "\t" << begin_gnode->get_output_shape(0)[0] << "\t"
                              << vec.size() << std::endl;
                    auto new_begin = std::make_shared<op::Constant>(element::i32, x_shape, vec);
                    begin_gnode = m_graph->add_node_and_edge(new_begin, GNodeVector{});
                }

                if (end_gnode->get_output_shape(0)[0] < dims)
                {
                    vector<int32_t> vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int32_t>(end_gnode, &vec));
                    for (size_t i = vec.size(); i < dims; i++)
                        vec.push_back(1);
                    auto new_end_gnode = std::make_shared<op::Constant>(element::i32, x_shape, vec);
                    end_gnode = m_graph->add_node_and_edge(new_end_gnode, GNodeVector{});
                }

                if (strides_gnode->get_output_shape(0)[0] < dims)
                {
                    vector<int32_t> vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int32_t>(strides_gnode, &vec));
                    for (size_t i = vec.size(); i < dims; i++)
                        vec.push_back(1);
                    auto new_strides_gnode =
                        std::make_shared<op::Constant>(element::i32, x_shape, vec);
                    strides_gnode = m_graph->add_node_and_edge(new_strides_gnode, GNodeVector{});
                }

                auto generic_gnode = m_graph->add_node_and_edge(
                    generic_op, {x_const_gnode, begin_gnode, end_gnode, strides_gnode, grad_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};

                return ret;
            }

            NamedNodeVector TranslateTileOp(const tensorflow::NodeDef& node,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                /*
                This operation creates a new tensor by replicating input multiples times.
                The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
                and the values of input are replicated multiples[i] times along the 'i'th dimension.
                For example, tiling [a b c d] by [2] produces [a b c d a b c d].
                */
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto multiples_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::vector<int64> in_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<int64>(multiples_gnode, &in_value))
                    << "TileOp currently do not support dynamic tensor shape";
                auto input_shape = multiples_gnode->get_shape();
                auto const_op = std::make_shared<op::Constant>(element::i64, input_shape, in_value);
                auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));
                nnfusion::op::OpConfig::any myConfig;
                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {input_gnode, const_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateUnsortedSegmentSumOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto seg_id_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto seg_num_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<int> in_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<int>(seg_num_gnode, &in_value))
                    << "We only accept the sgements number as Constant.";
                auto const_op = std::make_shared<op::Constant>(
                    element::i32, seg_num_gnode->get_shape(), in_value);
                auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));
                nnfusion::op::OpConfig::any myConfig;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode = m_graph->add_node_and_edge(
                    generic_op, {input_gnode, seg_id_gnode, const_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateSoftmaxOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto input_shape = input_gnode->get_shape();
                nnfusion::AxisSet ng_axes_softmax;
                auto rank = input_shape.size();
                NNFUSION_CHECK(rank >= 1) << "TF Softmax logits must be >=1 dimension";
                ng_axes_softmax.insert(rank - 1);

                if (!FLAGS_fantares_mode)
                {
                    auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                    softmax_op->set_name(node.name());
                    auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {input_gnode});
                    NamedNodeVector ret{{node.name(), softmax_gnode}};
                    return ret;
                }
                else
                {
                    auto softmax_gnode = TranslateSoftmaxToBasicOp(
                        input_gnode, ng_axes_softmax, node.name(), m_graph);
                    NamedNodeVector ret{{node.name(), softmax_gnode}};
                    return ret;
                }
            }

            NamedNodeVector TranslateAssertOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto constant_op = std::make_shared<nnfusion::op::Constant>(
                    nnfusion::element::i32, nnfusion::Shape{}, std::vector<int>{0});
                constant_op->set_name(node.name());

                auto constant_gnode = m_graph->add_node_and_edge(constant_op, GNodeVector());
                m_graph->add_control_edge(input_gnode, constant_gnode);
                NamedNodeVector ret{{node.name(), constant_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateReverseSequenceOp(const tensorflow::NodeDef& node,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto seq_length = GetInputNode(all_ng_nodes, node, 1);
                int64_t seq_dim, batch_dim;
                bool status = GetNodeAttr(node.attr(), "seq_dim", seq_dim);
                NNFUSION_CHECK(status);
                status = GetNodeAttr(node.attr(), "batch_dim", batch_dim);
                NNFUSION_CHECK(status);
                // Get the seq_lengths:vector<int64> from RS op
                nnfusion::op::OpConfig::any myConfig;
                myConfig["seq_axis"] = seq_dim;
                myConfig["batch_axis"] = batch_dim;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {input_gnode, seq_length});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateScatterOp(const tensorflow::NodeDef& node,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto ref = GetInputNode(all_ng_nodes, node, 0);
                auto indices = GetInputNode(all_ng_nodes, node, 1);
                auto updates = GetInputNode(all_ng_nodes, node, 2);

                //1. Check whether ref is variable
                // NNFUSION_CHECK(!ref->is_constant()) << "ScatterSub will only write back to variable.";

                nnfusion::op::OpConfig::any myConfig;
                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);
                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {ref, indices, updates});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateSelectOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input1_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto input2_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto input3_gnode = GetInputNode(all_ng_nodes, node, 2);

                NNFUSION_CHECK(input2_gnode->get_shape() == input3_gnode->get_shape())
                    << "Input tensors 2 and 3 should have same shape";

                auto input1_shape = input1_gnode->get_shape();
                auto input2_shape = input2_gnode->get_shape();

                auto input1_rank = input1_shape.size();
                auto input2_rank = input2_shape.size();

                NNFUSION_CHECK(((input1_shape == input2_shape) ||
                                ((input1_rank == 1) && (input2_rank > input1_rank) &&
                                 (input2_shape[0] == input1_shape[0]))))
                    << "Input tensor may have the same shape as condition. If condition is "
                    << "rank 1, input may have higher rank, but its first dimension must "
                    << "match the size of condition.";

                int length = 0;
                // shared_ptr<ngraph::Node> ng_input_new;

                // If input tensor has higher rank than condiiton, length will be > 0.
                length = input2_rank - input1_rank;

                if (length != 0)
                {
                    // Condition tensor will be modified to align the condition tensor
                    // shape with input tensor shape index and fill the rest of the vector
                    // with
                    // 1s
                    // Eg: condition tensor [7], input tensor [7, 3, 2, 1]
                    // After Reshape, condition tensor will be [7, 1 ,1 ,1] for auto
                    // broadcast.

                    std::vector<size_t> tmp_vector((input2_rank), 1);
                    tmp_vector[0] = input1_shape[0];

                    auto reshape_input1_op =
                        std::make_shared<op::Reshape>(nnfusion::AxisVector{0}, tmp_vector);
                    input1_gnode = m_graph->add_node_and_edge(reshape_input1_op, {input1_gnode});
                }

                std::tie(input1_gnode, input2_gnode) =
                    graph::numpy_broadcast(std::make_pair(input1_gnode, input2_gnode), m_graph);
                std::tie(input2_gnode, input3_gnode) =
                    graph::numpy_broadcast(std::make_pair(input2_gnode, input3_gnode), m_graph);

                auto select_op = std::make_shared<op::Select>();
                select_op->set_name(node.name());

                auto select_gnode = m_graph->add_node_and_edge(
                    select_op, {input1_gnode, input2_gnode, input3_gnode});

                NamedNodeVector ret{{node.name(), select_gnode}};
                return ret;
            }

            NamedNodeVector
                TranslateBroadcastGradientArgsOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                std::vector<BCast::Vec> shapes;
                GNodeVector input_gnodes;
                for (int i = 0; i < 2; ++i)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node, i);
                    input_gnodes.push_back(input_gnode);
                    auto input_shape = input_gnode->get_shape();
                    NNFUSION_CHECK(input_shape.size() == 1) << "input" << i << "must be a vector";
                    std::vector<int64> in_value;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int64>(input_gnode, &in_value));

                    BCast::Vec vec;
                    for (int64 i = 0; i < shape_size(input_shape); ++i)
                    {
                        vec.push_back(in_value[i]);
                    }
                    shapes.push_back(vec);
                }

                BCast bcast(shapes[0], shapes[1]);
                NNFUSION_CHECK(bcast.IsValid());
                // <<
                // "Incompatible shapes: [" << str_util::Join(shapes[0], ","),
                // "] vs. [", str_util::Join(shapes[1], ","), "]"));
                const BCast::Vec& out0 = bcast.grad_x_reduce_idx();
                const BCast::Vec& out1 = bcast.grad_y_reduce_idx();

                // todo: name???
                auto out_node_0_op = std::make_shared<op::Constant>(
                    nnfusion::element::i64, nnfusion::Shape({out0.size()}), out0);
                out_node_0_op->set_name(node.name() + "x");

                auto out_node_0_gnode = m_graph->add_node_and_edge(out_node_0_op, GNodeVector());
                m_graph->add_control_edge(input_gnodes[0], out_node_0_gnode);

                auto out_node_1_op = std::make_shared<op::Constant>(
                    nnfusion::element::i64, nnfusion::Shape({out1.size()}), out1);
                out_node_1_op->set_name(node.name() + "y");
                auto out_node_1_gnode = m_graph->add_node_and_edge(out_node_1_op, GNodeVector());
                m_graph->add_control_edge(input_gnodes[1], out_node_1_gnode);

                NamedNodeVector ret{{node.name(), out_node_0_gnode},
                                    {node.name(), out_node_1_gnode}};

                return ret;
            }

            NamedNodeVector TranslateFloorModOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input1_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto input2_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::tie(input1_gnode, input2_gnode) =
                    graph::numpy_broadcast(std::make_pair(input1_gnode, input2_gnode), m_graph);

                auto div_op = std::make_shared<op::Divide>();
                auto div_gnode = m_graph->add_node_and_edge(div_op, {input1_gnode, input2_gnode});

                auto floordiv_op = std::make_shared<op::Floor>();
                auto floordiv_gnode = m_graph->add_node_and_edge(floordiv_op, {div_gnode});

                auto multiply_op = std::make_shared<op::Multiply>();
                auto multiply_gnode =
                    m_graph->add_node_and_edge(multiply_op, {floordiv_gnode, input2_gnode});

                auto floormod_op = std::make_shared<op::Subtract>();
                floormod_op->set_name(node.name());
                auto floormod_gnode =
                    m_graph->add_node_and_edge(floormod_op, {input1_gnode, multiply_gnode});

                NamedNodeVector ret{{node.name(), floormod_gnode}};
                return ret;
            }
            NamedNodeVector
                TranslateDynamicStitchOp(const tensorflow::NodeDef& node,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                int32 num_partitions;
                GNodeVector input_gnodes;
                bool status = GetNodeAttr(node.attr(), "N", num_partitions);
                NNFUSION_CHECK(status);

                for (int i = 0; i < num_partitions * 2; i++)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node, i);
                    auto input_shape = input_gnode->get_shape();

                    if (i < num_partitions)
                    {
                        std::vector<int32> in_value;
                        NNFUSION_CHECK(GetValueFromNGraphOp<int32>(input_gnode, &in_value))
                            << "DynamicStitch currently do not support dynamic tensor shape";
                        auto const_op =
                            std::make_shared<op::Constant>(element::i32, input_shape, in_value);
                        auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));
                        input_gnodes.push_back(const_gnode);
                    }
                    else
                    {
                        input_gnodes.push_back(input_gnode);
                    }
                }

                nnfusion::op::OpConfig::any myConfig;
                myConfig["N"] = num_partitions;

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_gnodes});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            // Computes the gradient for tanh of 'x' w.r.t its input
            // grad = dy * (1 - y * y)
            // where y = tanh(x) and dy is the corresponding input gradient
            NamedNodeVector TranslateTanhGradOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 1);

                auto et = input_gnode->get_element_type();
                auto input_shape = input_gnode->get_shape();

                auto sq_op = std::make_shared<op::Multiply>();
                auto sq_gnode = m_graph->add_node_and_edge(sq_op, {input_gnode, input_gnode});

                std::vector<std::string> const_values(nnfusion::shape_size(input_shape), "1");

                auto const_op = std::make_shared<op::Constant>(et, input_shape, const_values);
                auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));

                auto sub_op = std::make_shared<op::Subtract>();
                auto sub_gnode = m_graph->add_node_and_edge(sub_op, {const_gnode, sq_gnode});

                auto multiply_op = std::make_shared<op::Multiply>();
                multiply_op->set_name(node.name());

                auto multiply_gnode =
                    m_graph->add_node_and_edge(multiply_op, {delta_gnode, sub_gnode});
                NamedNodeVector ret{{node.name(), multiply_gnode}};
                return ret;
            }

            NamedNodeVector TranslateFloorDivOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto rhs_gnode = GetInputNode(all_ng_nodes, node, 1);

                std::tie(lhs_gnode, rhs_gnode) =
                    graph::numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                auto div_op = std::make_shared<op::Divide>();
                auto div_gnode = m_graph->add_node_and_edge(div_op, {lhs_gnode, rhs_gnode});

                auto floor_op = std::make_shared<op::Floor>();
                floor_op->set_name(node.name());

                auto floor_gnode = m_graph->add_node_and_edge(floor_op, {div_gnode});

                NamedNodeVector ret{{node.name(), floor_gnode}};
                return ret;
            }

            NamedNodeVector TranslateShapeOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto& shape = input_gnode->get_shape();

                tensorflow::DataType dtype;
                bool status = GetNodeAttr(node.attr(), "out_type", dtype);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et;
                status = TFDataTypeToNNFusionElementType(dtype, &nnfusion_et);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(nnfusion_et == nnfusion::element::i32 ||
                               nnfusion_et == nnfusion::element::i64);

                nnfusion::Shape output_shape(1, shape.size());
                auto shape_op =
                    std::make_shared<op::Constant>(nnfusion_et, output_shape, shape); // TODO
                auto shape_gnode = m_graph->add_node_and_edge(shape_op, GNodeVector({}));
                NamedNodeVector ret{{node.name(), shape_gnode}};
                return ret;
            }

            NamedNodeVector TranslateUniqueOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto& input_shape = input_gnode->get_shape();
                NNFUSION_CHECK(nnfusion::is_vector(input_shape)) << "unique expects a 1D vector.";

                tensorflow::DataType out_idx;
                auto status = GetNodeAttr(node.attr(), "out_idx", out_idx);
                NNFUSION_CHECK(status);
                nnfusion::element::Type nnfusion_et_idx;
                status = TFDataTypeToNNFusionElementType(out_idx, &nnfusion_et_idx);
                NNFUSION_CHECK(status);
                NNFUSION_CHECK(nnfusion_et_idx == nnfusion::element::i32 ||
                               nnfusion_et_idx == nnfusion::element::i64);

                std::vector<int64> input_vec;
                status = GetValueFromNGraphOp<int64>(input_gnode, &input_vec);
                NNFUSION_CHECK(status);

                NNFUSION_CHECK(input_vec.size() <= std::numeric_limits<size_t>::max())
                    << "unique does not support input tensors larger than "
                    << std::numeric_limits<size_t>::max() << " elements";

                size_t N = input_vec.size();
                std::vector<int64> out_y;
                std::unordered_map<int64, int64> uniq;
                uniq.reserve(2 * N);
                std::vector<int64> idx_vec(N);
                for (size_t i = 0, j = 0; i < N; ++i)
                {
                    auto it = uniq.insert(std::make_pair(input_vec[i], j));
                    idx_vec[i] = it.first->second;
                    if (it.second)
                    {
                        ++j;
                    }
                }
                size_t uniq_size = uniq.size();
                std::vector<int64> y_vec(uniq_size);
                for (auto it : uniq)
                {
                    y_vec[it.second] = it.first;
                }

                auto y_op = std::make_shared<op::Constant>(
                    input_gnode->get_element_type(), nnfusion::Shape{uniq_size}, y_vec);
                y_op->set_name(node.name() + "y");
                auto y_gnode = m_graph->add_node_and_edge(y_op, GNodeVector({}));

                auto idx_op = std::make_shared<op::Constant>(nnfusion_et_idx, input_shape, idx_vec);
                idx_op->set_name(node.name() + "idx");
                auto idx_gnode = m_graph->add_node_and_edge(idx_op, GNodeVector({}));

                NamedNodeVector ret{{node.name(), y_gnode}, {node.name(), idx_gnode}};
                return ret;
            }

            NamedNodeVector TranslateUnpackOp(const tensorflow::NodeDef& node,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                nnfusion::Shape input_shape = input_gnode->get_shape();
                int rank = input_shape.size();

                // axis : Dimension along which to unpack.
                int32 axis;
                bool status = GetNodeAttr(node.attr(), "axis", axis);
                NNFUSION_CHECK(status);
                axis = axis + (axis < 0 ? rank : 0);

                // num : value of input_shape[axis]
                int32 num;
                status = GetNodeAttr(node.attr(), "num", num);
                NNFUSION_CHECK(status);

                std::vector<size_t> lower;
                std::vector<size_t> upper;
                nnfusion::Shape output_shape;
                for (int i = 0; i < rank; ++i)
                {
                    lower.push_back(0);
                    upper.push_back(input_shape[i]);
                    if (i != axis)
                    {
                        output_shape.push_back(input_shape[i]);
                    }
                }
                std::vector<size_t> shape_dimensions(input_shape.size());
                std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);

                int cursor = 0;
                NamedNodeVector ret;

                for (size_t i = 0; i < num; ++i)
                {
                    lower[axis] = cursor++;
                    upper[axis] = cursor;
                    auto slice_op = std::make_shared<op::Slice>(lower, upper);
                    slice_op->set_name(node.name() + "_slice" + std::to_string(i));
                    auto slice_gnode = m_graph->add_node_and_edge(slice_op, {input_gnode});

                    auto reshape_op = std::make_shared<op::Reshape>(shape_dimensions, output_shape);
                    reshape_op->set_name(node.name() + "_reshape_" + std::to_string(i));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {slice_gnode});

                    ret.push_back({node.name(), reshape_gnode});
                }
                return ret;
            }

            NamedNodeVector
                TranslateApplyGradientDescentOp(const tensorflow::NodeDef& node,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                auto alpha_gnode = GetInputNode(all_ng_nodes, node, 1);
                auto delta_gnode = GetInputNode(all_ng_nodes, node, 2);

                std::vector<float> alpha_value;
                NNFUSION_CHECK(GetValueFromNGraphOp<float>(alpha_gnode, &alpha_value))
                    << "We only accept the alpha as Constant.";
                NNFUSION_CHECK(alpha_value.size() == 1);

                nnfusion::op::OpConfig::any config;
                config["learning_rate"] = alpha_value[0];

                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), config);
                auto generic_gnode =
                    m_graph->add_node_and_edge(generic_op, {input_gnode, delta_gnode});
                NamedNodeVector ret{{node.name(), generic_gnode}};
                return ret;
            }

            NamedNodeVector TranslateZerosLikeOp(const tensorflow::NodeDef& node,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(all_ng_nodes, node, 0);
                shared_ptr<op::Constant> zeros;
                if (input_gnode->get_element_type() == nnfusion::element::i32)
                {
                    std::vector<int32_t> vec(1, 0);
                    zeros = std::make_shared<op::Constant>(
                        input_gnode->get_element_type(), input_gnode->get_output_shape(0), vec);
                }
                else if (input_gnode->get_element_type() == nnfusion::element::i64)
                {
                    std::vector<int64_t> vec(1, 0);
                    zeros = std::make_shared<op::Constant>(
                        input_gnode->get_element_type(), input_gnode->get_output_shape(0), vec);
                }
                if (input_gnode->get_element_type() == nnfusion::element::f32)
                {
                    std::vector<float> vec(1, 0.0);
                    zeros = std::make_shared<op::Constant>(
                        input_gnode->get_element_type(), input_gnode->get_output_shape(0), vec);
                }
                else
                    NNFUSION_CHECK_FAIL() << "Unsupported datatype.";
                auto zeros_gnode = m_graph->add_node_and_edge(zeros, GNodeVector{});
                NamedNodeVector ret{{node.name(), zeros_gnode}};
                return ret;
            }

            NamedNodeVector TranslateConcatOffsetOp(const tensorflow::NodeDef& node,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto concat_dim = GetInputNode(all_ng_nodes, node, 0);
                auto shape0 = GetInputNode(all_ng_nodes, node, 1);
                auto shape1 = GetInputNode(all_ng_nodes, node, 2);
                auto allinput = GetAllInputNode(all_ng_nodes, node);
                NNFUSION_CHECK(allinput.size() == 3) << "ConcatOffsetOp only support two shapes.";

                nnfusion::op::OpConfig::any config;
                auto generic_op =
                    std::make_shared<nnfusion::op::GenericOp>(node.name(), node.op(), config);

                auto _gnode =
                    m_graph->add_node_and_edge(generic_op, GNodeVector{concat_dim, shape0, shape1});
                NamedNodeVector ret{{node.name(), _gnode}};
                return ret;
            }

            const static std::map<const std::string, ConvertFunc> TRANSLATE_OP_MAP{
                {"Abs", TranslateUnaryOp<op::Abs>},
                {"Add", TranslateBinaryOp<op::Add>},
                {"AddV2", TranslateBinaryOp<op::Add>},
                {"AddN", TranslateAddNOp},
                {"All", TranslateAllOp},
                {"ApplyGradientDescent", TranslateApplyGradientDescentOp},
                {"ApplyMomentum", TranslateApplyMomentumOp},
                {"SparseApplyMomentum", TranslateSparseApplyMomentumOp},
                {"Assert", TranslateAssertOp},
                {"AvgPool", TranslateAvgPoolOp},
                {"BatchMatMul", TranslateBatchMatMulOp},
                {"BatchMatMulV2", TranslateBatchMatMulOp},
                {"BiasAdd", TranslateBiasAddOp},
                {"BroadcastGradientArgs", TranslateBroadcastGradientArgsOp},
                {"BiasAddGrad", TranslateBiasAddGradOp},
                {"Cast", TranslateCastOp},
                {"Const", TranslateConstOp},
                {"Conv2D", TranslateConv2DOp},
                {"ConcatV2", TranslateConcatV2Op},
                {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
                {"DivNoNan", TranslateBinaryOp<op::DivNoNan>},
                {"DynamicStitch", TranslateDynamicStitchOp},
                {"Equal", TranslateBinaryOp<op::Equal>},
                {"Erf", TranslateUnaryOp<op::Erf>},
                {"NotEqual", TranslateBinaryOp<op::NotEqual>},
                {"Exp", TranslateUnaryOp<op::Exp>},
                {"ExpandDims", TranslateExpandDimsOp},
                {"Fill", TranslateFillOp},
                {"FloorMod", TranslateFloorModOp},
                {"FloorDiv", TranslateFloorDivOp},
                {"FusedBatchNorm", TranslateFusedBatchNormOp},
                {"FusedBatchNormV2", TranslateFusedBatchNormOp},
                {"FusedBatchNormV3", TranslateFusedBatchNormOp},
                {"SpaceToDepth", TranslateSpaceToDepthOp},
                {"GatherV2", TranslateGatherV2Op},
                {"Greater", TranslateBinaryOp<op::Greater>},
                {"Identity", TranslateIdentityOp},
                {"InvertPermutation", TranslateInvertPermutationOp},
                {"LessEqual", TranslateBinaryOp<op::LessEq>},
                {"GreaterEqual", TranslateBinaryOp<op::GreaterEq>},
                {"Log", TranslateUnaryOp<op::Log>},
                {"MatMul", TranslateMatMulOp},
                {"Maximum", TranslateBinaryOp<op::Maximum>},
                {"MaxPool", TranslateMaxPoolOp},
                {"Mean", TranslateMeanOp},
                {"Mul", TranslateBinaryOp<op::Multiply>},
                {"Multiply", TranslateBinaryOp<op::Multiply>},
                {"Neg", TranslateUnaryOp<op::Negative>},
                {"NoOp", TranslateNoOp},
                {"OneHot", TranslateOneHotOp},
                {"Pack", TranslatePackOp},
                {"Pad", TranslatePadOp},
                {"PadV2", TranslatePadV2Op},
                {"Placeholder", TranslateTensorOp<op::Parameter>},
                {"PreventGradient", TranslateIdentityOp},
                {"Pow", TranslateBinaryOp<op::Power>},
                {"Range", TranslateRangeOp},
                {"Any", TranslateReduceAnyOp},
                {"Relu", TranslateUnaryOp<op::Relu>},
                {"Relu6", TranslateUnaryOp<op::Relu6>},
                {"ReluGrad", TranslateReluGradOp},
                {"Relu6Grad", TranslateRelu6GradOp},
                {"Reshape", TranslateReshapeOp},
                {"Rsqrt", TranslateUnaryOp<op::Rsqrt>},
                {"Sqrt", TranslateUnaryOp<op::Sqrt>},
                {"RsqrtGrad", TranslateRsqrtGradOp},
                {"RealDiv", TranslateBinaryOp<op::Divide>},
                {"ReverseSequence", TranslateReverseSequenceOp},
                {"ScatterSub", TranslateScatterOp},
                {"ScatterAdd", TranslateScatterOp},
                {"ScatterMin", TranslateScatterOp},
                {"ScatterMax", TranslateScatterOp},
                {"Select", TranslateSelectOp},
                {"Sigmoid", TranslateUnaryOp<op::Sigmoid>},
                {"Sign", TranslateUnaryOp<op::Sign>},
                {"Slice", TranslateSliceOp},
                {"Softmax", TranslateSoftmaxOp},
                {"Split", TranslateSplitOp},
                {"SplitV", TranslateSplitVOp},
                {"SquaredDifference", TranslateSquaredDifferenceOp},
                {"Squeeze", TranslateSqueezeOp},
                {"StopGradient", TranslateIdentityOp},
                {"StridedSlice", TranslateStridedSliceOp},
                {"SparseSoftmaxCrossEntropyWithLogits",
                 TranslateSparseSoftmaxCrossEntropyWithLogitsOp},
                {"StridedSliceGrad", TranslateStridedSliceGradOp},
                //{"", TranslateStopGradientOp},
                {"Sub", TranslateBinaryOp<op::Subtract>},
                {"Sum", TranslateSumOp},
                {"Tanh", TranslateUnaryOp<op::Tanh>},
                {"TanhGrad", TranslateTanhGradOp},
                {"Tile", TranslateTileOp},
                //{"", TranslateTransposeOp},
                {"Unique", TranslateUniqueOp},
                {"UnsortedSegmentSum", TranslateUnsortedSegmentSumOp},
                // {"VariableV2", TranslateTensorOp<op::Variable>},
                {"VariableV2", TranslateTensorOp<op::Parameter>},
                {"Transpose", TranslateTransposeToReshapeOp},
                {"Square", TranslateUnaryOp<op::Square>},
                {"Shape", TranslateShapeOp},
                {"ConcatOffset", TranslateConcatOffsetOp},
                {"ZerosLike", TranslateZerosLikeOp},
                {"SigmoidGrad", TranslateSigmoidGradOp},
                {"Assign", TranslateAssignOp},
                {"AssignSub", TranslateAssignSubOp},
                {"ApplyAdam", TranslateApplyAdamOp},
                {"Unpack", TranslateUnpackOp}};

            bool check_model_availability(const tensorflow::GraphDef* graph_proto)
            {
                auto op_configs = op::get_op_configs();
                const size_t num_nodes = graph_proto->node_size();
                std::unordered_set<std::string> unknown_ops;
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    std::string op_type = graph_proto->node(n).op();
                    if (TRANSLATE_OP_MAP.find(op_type) == TRANSLATE_OP_MAP.end() &&
                        op_configs.find(op_type) == op_configs.end())
                    {
                        unknown_ops.insert(op_type);
                    }
                }
                if (unknown_ops.size() > 0)
                {
                    for (auto& op_type : unknown_ops)
                    {
                        NNFUSION_LOG(ERROR) << "Unsupported tf op: " << op_type;
                    }
                    return false;
                }
                return true;
            }
            struct InputInfo
            {
                explicit InputInfo(const std::string& node_name,
                                   std::shared_ptr<nnfusion::graph::GNode> n,
                                   int i)
                    : name(node_name)
                    , node(n)
                    , index(i)
                {
                }
                std::string name;
                std::shared_ptr<nnfusion::graph::GNode> node;
                int index;
            };

            GraphConvert::GraphConvert(const tensorflow::GraphDef& proto)
                : tf_graph_proto{&proto}
            {
                NNFUSION_LOG(INFO) << "Converting Tensorflow Graph";

                NNFUSION_CHECK(check_model_availability(tf_graph_proto));

                m_graph = std::make_shared<nnfusion::graph::Graph>();

                generate_topology();

                std::vector<InputInfo> inputs;
                while (!tf_topology_.empty())
                {
                    uint32_t node_idx = tf_topology_.front();
                    tf_topology_.pop();
                    inputs.clear();
                    const auto& node_proto = proto.node(node_idx);
                    bool in_control_dependence = false;
                    for (auto& input : node_proto.input())
                    {
                        TensorId input_tensor(ParseTensorName(input));
                        int src_index = input_tensor.second;

                        std::shared_ptr<nnfusion::graph::GNode> src_node;

                        auto iter = m_node_map.find(input_tensor.first);
                        NNFUSION_CHECK(iter != m_node_map.end())
                            << "Node " << node_proto.name()
                            << " has Un-Converted input node: " << input_tensor.first;
                        if (src_index == nnfusion::graph::Graph::kControlSlot)
                        {
                            in_control_dependence = true;
                            if (iter->second.size() > 0)
                            {
                                src_node = iter->second.at(0);
                                inputs.emplace_back(input_tensor.first, src_node, -1);
                            }
                        }
                        else
                        {
                            NNFUSION_CHECK(!in_control_dependence)
                                << "Control dependencies must come after regular "
                                   "dependencies.";
                            src_node = iter->second.at(src_index);
                            inputs.emplace_back(input_tensor.first, src_node, 0);
                        }
                    }

                    auto results = convert_node(node_proto);
                    m_node_map[node_proto.name()] = {};
                    for (auto& name_gnode_pair : results)
                    {
                        auto gnode = name_gnode_pair.second;
                        m_node_map[name_gnode_pair.first].push_back(gnode);

                        // add control edge
                        for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++)
                        {
                            NNFUSION_CHECK_NOT_NULLPTR(inputs[input_idx].node)
                                << "Back edge is not supported now.";

                            if (inputs[input_idx].index == nnfusion::graph::Graph::kControlSlot)
                            {
                                m_graph->add_control_edge(inputs[input_idx].node, gnode);
                            }
                            else
                            {
                                // normal edge, do nothing
                            }
                        }
                        if (gnode->get_name() != name_gnode_pair.first)
                        {
                            //NNFUSION_CHECK(!(*gnode)["Alias"].is_valid())
                            //    << "node " << gnode->get_name() << " has more than one alias.\nThe tensorflow node is : \n" << node_proto.DebugString();
                            (*gnode)["Alias"] = name_gnode_pair.first;
                        }

                        if (tf_output_name_.find(node_proto.name()) != tf_output_name_.end())
                        {
                            m_graph_outputs.emplace_back(gnode);
                        }
                    }

                    for (size_t i = 0; i < tf_node_outputs_[node_idx].size(); ++i)
                    {
                        const int output = tf_node_outputs_[node_idx][i];
                        tf_pending_counts_[output]--;
                        if (tf_pending_counts_[output] == 0)
                        {
                            tf_topology_.push(output);
                        }
                    }
                }

                m_graph->set_outputs(m_graph_outputs);
                m_graph->set_default_parameters();
                NNFUSION_LOG(INFO) << "convert graph done";
            }

            void GraphConvert::generate_topology()
            {
                const size_t num_nodes = tf_graph_proto->node_size();
                std::unordered_map<std::string, uint32_t> tensorflow_name2nodeIdx_map;
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    tensorflow_name2nodeIdx_map[tf_graph_proto->node(n).name()] = n;
                }

                tf_pending_counts_.reserve(num_nodes);
                tf_node_outputs_.resize(num_nodes);
                for (size_t n = 0; n < num_nodes; ++n)
                {
                    const auto& node_proto = tf_graph_proto->node(n);
                    int pending_count = node_proto.input_size();
                    for (size_t i = 0; i < node_proto.input_size(); ++i)
                    {
                        std::string input_name = node_proto.input(i);
                        TensorId input_tensor(ParseTensorName(input_name));

                        auto iter = tensorflow_name2nodeIdx_map.find(input_tensor.first);
                        NNFUSION_CHECK(iter != tensorflow_name2nodeIdx_map.end())
                            << "Node " << node_proto.name()
                            << " has Unknown input node: " << input_name;

                        tf_node_outputs_[iter->second].push_back(n);
                    }
                    if (pending_count == 0)
                    {
                        tf_topology_.push(n);
                    }
                    tf_pending_counts_.push_back(pending_count);
                }

                for (size_t n = 0; n < num_nodes; ++n)
                {
                    if (tf_node_outputs_[n].size() == 0)
                    {
                        tf_output_name_.insert(tf_graph_proto->node(n).name());
                    }
                }
            }

            NamedNodeVector GraphConvert::convert_node(const tensorflow::NodeDef& node)
            {
                NamedNodeVector ret;
                auto func = TRANSLATE_OP_MAP.find(node.op());
                if (func != TRANSLATE_OP_MAP.end())
                {
                    ret = func->second(node, m_node_map, m_graph);
                }
                else
                {
                    ret = TranslateGenericNoAttrOp(node, m_node_map, m_graph);
                }
                return std::move(ret);
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
