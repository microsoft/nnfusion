//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include "graph_convert.hpp"
#include "../ops/const.hpp"
#include "nnfusion/common/axis_vector.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

#define TS_IMPORT_DEBUG 1

using namespace nnfusion::graph;
namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            template <typename T>
            GNodeVector TranslateUnaryOp(const TNodePtr n,
                                         NodeMap& tnode2gnodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto input_gnode = GetInputNode(tnode2gnodes, n, 0);
                auto op = std::make_shared<T>();
                op->set_name(n->output()->debugName());
                auto gnode = m_graph->add_node_and_edge(op, {input_gnode});
                return {gnode};
            }

            template <typename T>
            GNodeVector TranslateBinaryOp(const TNodePtr n,
                                          NodeMap& tnode2gnodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(tnode2gnodes, n, 0);
                auto rhs_gnode = GetInputNode(tnode2gnodes, n, 1);

                std::tie(lhs_gnode, rhs_gnode) =
                    numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);

                auto op = std::make_shared<T>();
                op->set_name(n->output()->debugName());
                auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});

                return {gnode};
            }

            GNodeVector TranslateDot(const TNodePtr n,
                                     NodeMap& tnode2gnodes,
                                     std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(tnode2gnodes, n, 0);
                auto rhs_gnode = GetInputNode(tnode2gnodes, n, 1);

                auto op = std::make_shared<op::Dot>();
                op->set_name(n->output()->debugName());
                auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});

                return {gnode};
            }

            GNodeVector TranslateMatMul(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                auto lhs_gnode = GetInputNode(tnode2gnodes, n, 0);
                auto rhs_gnode = GetInputNode(tnode2gnodes, n, 1);

                GNodeVector ret;
                auto lhs_rank = lhs_gnode->get_shape().size();
                auto rhs_rank = rhs_gnode->get_shape().size();
                if (rhs_rank <= 2)
                { // mat-vec, mat-mat, batch mat-mat
                    auto op = std::make_shared<op::Dot>();
                    op->set_name(n->output()->debugName());
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});
                    ret.push_back(gnode);
                }
                else if (lhs_rank == rhs_rank)
                { // batch mat-mat
                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["adj_x"]["b"] = false;
                    myConfig["adj_y"]["b"] = false;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        n->output()->debugName(),
                        "BatchMatMul", // select which existing kernels to use;
                        myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, {lhs_gnode, rhs_gnode});
                    ret.push_back(generic_gnode);
                }
                else
                {
                    // TODO: batch matmul with broadcast
                    // e.g [1000, 500, 100, 10] * [500, 10, 50]
                    // 1. broadcast [500, 10, 50] to [1000, 500, 10, 50]
                    // 2. batch matmul
                    NNFUSION_CHECK_FAIL() << "Not support matmul shape, lhs_rank: " << lhs_rank
                                          << ", rhs_rank" << rhs_rank;
                }
                return ret;
            }

            GNodeVector TranslateAdd(const TNodePtr n,
                                     NodeMap& tnode2gnodes,
                                     std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // aten::add accept 3 inputs(x, y, alpha), output = x + alpha * y
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto x = input_gnodes[0];
                auto y = input_gnodes[1];
                auto alpha = input_gnodes[2];

                auto cast_op = std::make_shared<op::Convert>(y->get_element_type());
                alpha = m_graph->add_node_and_edge(cast_op, {alpha});
                std::tie(y, alpha) = numpy_broadcast(std::make_pair(y, alpha), m_graph);

                auto add_right =
                    m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, alpha});

                std::tie(x, add_right) = numpy_broadcast(std::make_pair(x, add_right), m_graph);
                auto out_gnode =
                    m_graph->add_node_and_edge(std::make_shared<op::Add>(), {x, add_right});

                return {out_gnode};
            }

            GNodeVector TranslateT(const TNodePtr n,
                                   NodeMap& tnode2gnodes,
                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_dim = input_gnodes[0]->get_shape().size();
                AxisVector ng_axis_order(input_dim);
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                if (input_dim >= 2)
                {
                    std::swap(ng_axis_order[0], ng_axis_order[1]);
                }

                auto out_gnode = numpy_transpose(input_gnodes[0], ng_axis_order);
                m_graph->add_node(out_gnode);
                m_graph->add_edge(input_gnodes[0], 0, out_gnode, 0);

                return {out_gnode};
            }

            GNodeVector TranslateDim(const TNodePtr n,
                                     NodeMap& tnode2gnodes,
                                     std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_dim = input_gnodes[0]->get_shape().size();
                auto out_op = MakeConstOp({}, std::vector<int64>{static_cast<int64>(input_dim)});

                auto out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));
                return {out_gnode};
            }

            GNodeVector TranslateEq(const TNodePtr n,
                                    NodeMap& tnode2gnodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto lhs_gnode = input_gnodes[0];
                auto rhs_gnode = input_gnodes[1];
                if (lhs_gnode->get_element_type() != rhs_gnode->get_element_type())
                {
                    auto cast_op = std::make_shared<op::Convert>(lhs_gnode->get_element_type());
                    rhs_gnode = m_graph->add_node_and_edge(cast_op, {rhs_gnode});
                }
                std::tie(lhs_gnode, rhs_gnode) =
                    numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);
                GNodePtr out_gnode = m_graph->add_node_and_edge(std::make_shared<op::Equal>(),
                                                                {lhs_gnode, rhs_gnode});
                return {out_gnode};
            }

            GNodeVector Translate__isnot__(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                bool is_not = true;
                if (input_gnodes[0] == nullptr && input_gnodes[1] == nullptr)
                {
                    is_not = false;
                }
                else if (input_gnodes[0] == nullptr || input_gnodes[1] == nullptr)
                {
                    is_not = true;
                }
                else
                {
                    NNFUSION_CHECK_FAIL()
                        << "Currently only support __isnot__ compared with None type";
                }

                auto out_op = MakeConstOp(nnfusion::Shape{}, std::vector<bool>{is_not});

                auto out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));

                return {out_gnode};
            }

            GNodeVector TranslateIf(const TNodePtr n,
                                    NodeMap& tnode2gnodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto if_condition = GetConstValues<bool>(input_gnodes[0], 1)[0];

                GNodeVector out_gnodes;
                if (if_condition)
                {
                    out_gnodes = convert_block(n->blocks()[0], tnode2gnodes, m_graph);
                }
                else
                {
                    out_gnodes = convert_block(n->blocks()[1], tnode2gnodes, m_graph);
                }

                return out_gnodes;
            }

            GNodeVector TranslateConstant(const TNodePtr n,
                                          NodeMap& tnode2gnodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                if (!n->hasAttributeS("value"))
                {
                    // It's a torchscript "None" node, ideally there should be a "None" ngraph node, as a workaround, we return null ptr
                    return {nullptr};
                }
                auto out_op = MakeConstOp(n);
                GNodePtr out_gnode = nullptr;
                if (!isNoneConst(out_op))
                {
                    out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));
                }
                return {out_gnode};
            }

            GNodeVector TranslateReturn(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                return input_gnodes;
            }

            GNodeVector TranslateNo(const TNodePtr n,
                                    NodeMap& tnode2gnodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                return input_gnodes;
            }

            GNodeVector TranslateListConstruct(const TNodePtr n,
                                               NodeMap& tnode2gnodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                // TODO: empty list should be support, but currently we have no such type
                NNFUSION_CHECK(input_gnodes.size() > 0) << "ListConstruct has an empty inputs";
                auto output_type = n->output()->type()->kind();
                NNFUSION_CHECK(output_type == c10::TypeKind::ListType)
                    << "ListConstruct output must be list type";
                auto output_type_ptr = n->output()->type()->cast<c10::ListType>();
                auto input_type = output_type_ptr->getElementType();

                std::shared_ptr<nnfusion::op::Op> out_op = nullptr;
                // TODO: templated it
                switch (input_type->kind())
                {
                case c10::TypeKind::IntType:
                {
                    std::vector<int64> list_value;
                    const Shape op_shape = input_gnodes[0]->get_shape();
                    auto list_shape = op_shape;
                    list_shape.insert(list_shape.begin(), input_gnodes.size());

                    for (auto n : input_gnodes)
                    {
                        NNFUSION_CHECK(n->get_shape() == op_shape)
                            << "ListConstruct input must have a consistent shape";
                        auto op_value = GetConstValues<int64>(n);
                        list_value.insert(list_value.end(), op_value.begin(), op_value.end());
                    }

                    out_op = MakeConstOp(list_shape, list_value);
                    break;
                }
                case c10::TypeKind::BoolType:
                {
                    std::vector<bool> list_value;
                    const Shape op_shape = input_gnodes[0]->get_shape();
                    auto list_shape = op_shape;
                    list_shape.insert(list_shape.begin(), input_gnodes.size());

                    for (auto n : input_gnodes)
                    {
                        NNFUSION_CHECK(n->get_shape() == op_shape)
                            << "ListConstruct input must have a consistent shape";
                        auto op_value = GetConstValues<bool>(n);
                        list_value.insert(list_value.end(), op_value.begin(), op_value.end());
                    }

                    out_op = MakeConstOp(list_shape, list_value);
                    break;
                }
                default: NNFUSION_CHECK_FAIL() << "Unsupported data type";
                }

                auto out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));

                return {out_gnode};
            }

            GNodeVector TranslateConvolution2D(const TNodePtr n,
                                               NodeMap& tnode2gnodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // _convolution inputs: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();

                // std::cout << input_shape << std::endl;
                NNFUSION_CHECK(input_shape.size() == 4)
                    << "Convolution input must be 4-d tensor, but input dim is "
                    << input_shape.size();
                auto stride = GetConstValues<int64>(input_gnodes[3], 2);
                auto padding = GetConstValues<int64>(input_gnodes[4], 2);
                auto dilation = GetConstValues<int64>(input_gnodes[5], 2);
                auto transposed = GetConstValues<bool>(input_gnodes[6], 1)[0];
                NNFUSION_CHECK(transposed == false) << "Transposed convolution not supported";
                auto output_padding = GetConstValues<int64>(input_gnodes[7], 2);
                auto groups = GetConstValues<int64>(input_gnodes[8], 1);
                NNFUSION_CHECK(groups[0] == 1) << "Groups convolution not supported";

                auto conv_dim = input_shape.size() - 2;
                Strides ng_strides(stride.begin(), stride.end());
                Strides ng_dilations(dilation.begin(), dilation.end());
                CoordinateDiff ng_padding_below(conv_dim, padding[0]);
                CoordinateDiff ng_padding_above(conv_dim, padding[1]);

                auto conv_op = std::make_shared<op::Convolution>(
                    ng_strides, ng_dilations, ng_padding_below, ng_padding_above);

                auto conv_gnode =
                    m_graph->add_node_and_edge(conv_op, {input_gnodes[0], input_gnodes[1]});

                AxisSet broadcast_axes;
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    if (i != 1)
                    {
                        broadcast_axes.insert(i);
                    }
                }
                auto bias_broadcasted_op =
                    std::make_shared<op::Broadcast>(conv_gnode->get_shape(), broadcast_axes);

                auto bias_broadcasted_gnode =
                    m_graph->add_node_and_edge(bias_broadcasted_op, {input_gnodes[2]});

                auto add_op = std::make_shared<op::Add>();

                auto out_gnode =
                    m_graph->add_node_and_edge(add_op, {conv_gnode, bias_broadcasted_gnode});

                return {out_gnode};
            }

            GNodeVector TranslateMaxPool2D(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // MaxPool2D inputs: input, kernel_size, stride, padding, dilation, ceil_mode
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                NNFUSION_CHECK(input_shape.size() == 4)
                    << "MaxPool2D input must be 4-d tensor, but input dim is "
                    << input_shape.size();
                auto kernel_size = GetConstValues<int64>(input_gnodes[1], 2);
                auto stride = GetConstValues<int64>(input_gnodes[2], 2);
                auto padding = GetConstValues<int64>(input_gnodes[3], 2);
                auto dilation = GetConstValues<int64>(input_gnodes[4], 2);
                NNFUSION_CHECK(std::all_of(dilation.begin(), dilation.end(), [](int64 i) {
                    return i == 1;
                })) << "MaxPool2D dilation should be 1";
                // auto ceil_mode = GetConstValues<bool>(input_gnodes[5], 1)[0];
                // NNFUSION_CHECK(ceil_mode) << "MaxPool2D ceil_mode not supported";

                auto pad_dim = input_shape.size() - 2;
                Shape ng_kernel_shape(kernel_size.begin(), kernel_size.end());
                Strides ng_strides(stride.begin(), stride.end());
                Shape ng_padding_below(pad_dim, padding[0]);
                Shape ng_padding_above(pad_dim, padding[1]);

                auto maxpool_op = std::make_shared<nnfusion::op::MaxPool>(
                    ng_kernel_shape, ng_strides, ng_padding_below, ng_padding_above);
                auto maxpool_gnode = m_graph->add_node_and_edge(maxpool_op, {input_gnodes[0]});

                return {maxpool_gnode};
            }

            GNodeVector TranslateAdaptiveAvgPool2d(const TNodePtr n,
                                                   NodeMap& tnode2gnodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // AdaptiveAvgPool2d inputs: input, output_size
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                NNFUSION_CHECK(input_shape.size() == 4)
                    << "AdaptiveAvgPool2d input must be 4-d tensor, but input dim is "
                    << input_shape.size();
                auto output_size = GetConstValues<int64>(input_gnodes[1], 2);
                NNFUSION_CHECK(output_size[0] % input_shape[2] == 0 &&
                               output_size[1] % input_shape[3] == 0)
                    << "Currently, AdaptiveAvgPool2d input size must be a multiple of output size";

                auto multiple_h = output_size[0] / input_shape[2];
                auto multiple_w = output_size[1] / input_shape[3];

                auto pad_dim = input_shape.size() - 2;
                Shape ng_kernel_shape{multiple_h, multiple_w};
                Strides ng_strides{multiple_h, multiple_w};
                Shape ng_padding_below(pad_dim, 0);
                Shape ng_padding_above(pad_dim, 0);

                auto avgpool_op = std::make_shared<nnfusion::op::AvgPool>(
                    ng_kernel_shape, ng_strides, ng_padding_below, ng_padding_above);
                auto avgpool_gnode = m_graph->add_node_and_edge(avgpool_op, {input_gnodes[0]});

                return {avgpool_gnode};
            }

            GNodeVector TranslateSize(const TNodePtr n,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, axis
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto axis_vector = GetConstValues<int64>(input_gnodes[1], 1);
                auto input_dim = input_gnodes[0]->get_shape().at(axis_vector[0]);
                auto out_op = MakeConstOp(nnfusion::Shape{},
                                          std::vector<int64>{static_cast<int64>(input_dim)});

                auto out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));
                return {out_gnode};
            }

            GNodeVector TranslateInt(const TNodePtr n,
                                     NodeMap& tnode2gnodes,
                                     std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // Get and convert the first element to int
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);
                NNFUSION_CHECK(shape_size(input_gnodes[0]->get_shape()) == 1);

                auto cast_op = std::make_shared<op::Convert>(nnfusion::element::i64);
                auto out_gnode = m_graph->add_node_and_edge(cast_op, {input_gnodes[0]});

                return {out_gnode};
            }

            GNodeVector TranslateView(const TNodePtr n,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, shape
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto shape_vector = GetConstValues<int64>(input_gnodes[1]);
                NNFUSION_CHECK(std::count(shape_vector.begin(), shape_vector.end(), -1) <= 1)
                    << "Shape should have at most 1 dynamic dimension";

                size_t num_input_elements = nnfusion::shape_size(input_gnodes[0]->get_shape());

                // infer the dimension of -1
                auto dynamic_dim = shape_vector.end();
                size_t static_size = 1;
                for (auto it = shape_vector.begin(); it != shape_vector.end(); it++)
                {
                    if (*it == -1)
                    {
                        dynamic_dim = it;
                    }
                    else
                    {
                        static_size *= *it;
                    }
                }
                if (dynamic_dim == shape_vector.end())
                {
                    NNFUSION_CHECK(static_size == num_input_elements)
                        << "Reshape size doesn\'t match";
                }
                else
                {
                    NNFUSION_CHECK(num_input_elements % static_size == 0)
                        << "The product of static dims cannot be evenly divided by element number.";
                    *dynamic_dim = num_input_elements / static_size;
                }

                nnfusion::Shape ng_shape(shape_vector.begin(), shape_vector.end());

                nnfusion::AxisVector ng_axis_order(input_gnodes[0]->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto reshape_op = std::make_shared<nnfusion::op::Reshape>(ng_axis_order, ng_shape);
                // reshape_op->set_name(node.name());
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_gnodes[0]});
                return {reshape_gnode};
            }

            GNodeVector TranslateFlatten(const TNodePtr n,
                                         NodeMap& tnode2gnodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, start_dim, end_dim
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                auto start_dim = GetConstValues<int64>(input_gnodes[1], 1)[0];
                auto end_dim = GetConstValues<int64>(input_gnodes[2], 1)[0];

                if (end_dim < 0)
                {
                    end_dim += input_shape.size();
                }
                NNFUSION_CHECK(start_dim <= end_dim)
                    << "Start dim should be lower or equal to end dim.";
                std::vector<int64> shape_vector;
                for (auto i = 0; i < start_dim; i++)
                {
                    shape_vector.push_back(input_shape[i]);
                }
                shape_vector.push_back(-1);
                for (auto i = end_dim + 1; i < input_shape.size(); i++)
                {
                    shape_vector.push_back(input_shape[i]);
                }

                size_t num_input_elements = nnfusion::shape_size(input_gnodes[0]->get_shape());
                auto dynamic_dim = shape_vector.end();
                size_t static_size = 1;
                for (auto it = shape_vector.begin(); it != shape_vector.end(); it++)
                {
                    if (*it == -1)
                    {
                        dynamic_dim = it;
                    }
                    else
                    {
                        static_size *= *it;
                    }
                }

                if (dynamic_dim == shape_vector.end())
                {
                    NNFUSION_CHECK(static_size == num_input_elements)
                        << "Reshape size doesn\'t match";
                }
                else
                {
                    NNFUSION_CHECK(num_input_elements % static_size == 0)
                        << "The product of static dims cannot be evenly divided by element number.";
                    *dynamic_dim = num_input_elements / static_size;
                }

                nnfusion::Shape ng_shape(shape_vector.begin(), shape_vector.end());

                nnfusion::AxisVector ng_axis_order(input_gnodes[0]->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto reshape_op = std::make_shared<nnfusion::op::Reshape>(ng_axis_order, ng_shape);
                // reshape_op->set_name(node.name());
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_gnodes[0]});
                return {reshape_gnode};
            }

            GNodeVector TranslateAddmm(const TNodePtr n,
                                       NodeMap& tnode2gnodes,
                                       std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: bias, input, weight_t, beta, alpha
                // output = beta (bias) + alpha (input * weight_t)
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto bias = input_gnodes[0];
                auto input = input_gnodes[1];
                auto weight_t = input_gnodes[2];
                auto beta = input_gnodes[3];
                auto alpha = input_gnodes[4];
                NNFUSION_CHECK(beta->is_constant() &&
                               input_gnodes[3]->get_element_type() == element::i64)
                    << "Addmm beta must be int";
                NNFUSION_CHECK(alpha->is_constant() &&
                               input_gnodes[4]->get_element_type() == element::i64)
                    << "Addmm alpha must be int";

                auto cast_op = std::make_shared<op::Convert>(bias->get_element_type());
                beta = m_graph->add_node_and_edge(cast_op, {beta});
                std::tie(bias, beta) = numpy_broadcast(std::make_pair(bias, beta), m_graph);
                auto add_left =
                    m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {bias, beta});

                auto dot =
                    m_graph->add_node_and_edge(std::make_shared<op::Dot>(), {input, weight_t});
                cast_op = std::make_shared<op::Convert>(dot->get_element_type());
                alpha = m_graph->add_node_and_edge(cast_op, {alpha});
                std::tie(dot, alpha) = numpy_broadcast(std::make_pair(dot, alpha), m_graph);
                auto add_right =
                    m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {dot, alpha});

                std::tie(add_left, add_right) =
                    numpy_broadcast(std::make_pair(add_left, add_right), m_graph);

                auto gnode =
                    m_graph->add_node_and_edge(std::make_shared<op::Add>(), {add_left, add_right});

                return {gnode};
            }

            GNodeVector TranslateDropout(const TNodePtr n,
                                         NodeMap& tnode2gnodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, prob, train
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input = input_gnodes[0];
                auto prob = input_gnodes[1];
                auto train = input_gnodes[2];

                auto in_train = GetConstValues<bool>(train, 1)[0];
                NNFUSION_CHECK(in_train == false) << "Dropout train_mode must be false";

                return {input};
            }

            GNodeVector TranslatePermute(const TNodePtr n,
                                         NodeMap& tnode2gnodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, axis_order
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input = input_gnodes[0];

                auto input_shape = input->get_shape();
                auto axis_order = GetConstValues<int64>(input_gnodes[1]);

                size_t output_rank = input_shape.size();

                // Convert the values from the constant into an nnfusion::Shape, and
                // construct the axis order while we are at it.
                nnfusion::Shape output_shape(output_rank);

                for (size_t i = 0; i < output_rank; i++)
                {
                    output_shape[i] = input_shape[axis_order[i]];
                }

                nnfusion::AxisVector ng_axis_order;

                ng_axis_order.reserve(output_rank);

                for (int i = 0; i < output_rank; i++)
                {
                    ng_axis_order.push_back(axis_order[i]);
                }
                auto reshape_op =
                    std::make_shared<nnfusion::op::Reshape>(ng_axis_order, output_shape);
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input});

                return {reshape_gnode};
            }

            GNodeVector TranslateExpand(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, shape, implicit == false
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input = input_gnodes[0];

                auto input_shape = input->get_shape();
                size_t input_rank = input_shape.size();
                auto expand_shape = GetConstValues<int64>(input_gnodes[1]);

                size_t output_rank = expand_shape.size();

                NNFUSION_CHECK(output_rank >= input_rank)
                    << "the number of sizes provided (" << output_rank
                    << ") must be greater or equal to the number of dimensions in the tensor ("
                    << input_rank << ")";

                nnfusion::Shape shape(output_rank);
                for (size_t i = 0; i < output_rank; i++)
                {
                    int dim = i - output_rank + input_rank;
                    // for the new dimensions
                    if (dim < 0)
                    {
                        NNFUSION_CHECK(expand_shape[i] > 0)
                            << "For the new dimensions, the expand size should be greater than 1.";
                    }
                    else
                    {
                        // Passing -1 as the size for a dimension means not changing the size of that dimension.
                        if (expand_shape[i] == -1)
                        {
                            expand_shape[i] = input_shape[dim];
                        }
                        else if (expand_shape[i] != input_shape[dim])
                        {
                            NNFUSION_CHECK(input_shape[dim] == 1)
                                << "The expanded size of the tensor must match the existing size "
                                   "at non-singleton dimension.";
                        }
                    }
                    shape[i] = (size_t)expand_shape[i];
                }
                auto expand_shape_op =
                    std::make_shared<op::Constant>(element::i32, shape, std::vector<int>({1}));
                auto expand_shape_gnode =
                    m_graph->add_node_and_edge(expand_shape_op, GNodeVector({}));

                std::tie(input, expand_shape_gnode) =
                    numpy_broadcast(std::make_pair(input, expand_shape_gnode), m_graph);

                return {input};
            }

            GNodeVector TranslateSoftMax(const TNodePtr n,
                                         NodeMap& tnode2gnodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, dim, None
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto dim = GetConstValues<int64>(input_gnodes[1], 1)[0];
                dim += dim < 0 ? input_gnodes[0]->get_shape().size() : 0;
                nnfusion::AxisSet ng_axes_softmax;
                ng_axes_softmax.insert(dim);

                auto softmax_op = std::make_shared<op::Softmax>(ng_axes_softmax);
                auto softmax_gnode = m_graph->add_node_and_edge(softmax_op, {input_gnodes[0]});

                return {softmax_gnode};
            }
            GNodeVector TranslateNone(const TNodePtr n,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                return {nullptr};
            }

            GNodeVector TranslateZeros(const TNodePtr n,
                                       NodeMap& tnode2gnodes,
                                       std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: size, dtype, layout, device, pin_memory
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto shape = GetConstValues<int64>(input_gnodes[0]);
                auto dtype = GetConstValues<int64>(input_gnodes[1], 1);

                auto ele_num = nnfusion::shape_size(shape);
                auto ele_shape = nnfusion::Shape(shape.begin(), shape.end());
                auto ele_type = static_cast<c10::ScalarType>(dtype[0]);

                nnfusion::element::Type ng_type;
                NNFUSION_CHECK(ScalarTypeToNGraphElementType(ele_type, &ng_type));

                auto values = std::vector<string>(ele_num, "0");
                auto const_op = make_shared<op::Constant>(ng_type, ele_shape, values);
                GNodePtr out_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));

                return {out_gnode};
            }

            GNodeVector TranslateOnes(const TNodePtr n,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: size, dtype, layout, device, pin_memory
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto shape = GetConstValues<int64>(input_gnodes[0]);
                auto dtype = GetConstValues<int64>(input_gnodes[1], 1);

                auto ele_num = nnfusion::shape_size(shape);
                auto ele_shape = nnfusion::Shape(shape.begin(), shape.end());
                auto ele_type = static_cast<c10::ScalarType>(dtype[0]);

                nnfusion::element::Type ng_type;
                NNFUSION_CHECK(ScalarTypeToNGraphElementType(ele_type, &ng_type));

                auto values = std::vector<string>(ele_num, "1");
                auto const_op = make_shared<op::Constant>(ng_type, ele_shape, values);
                GNodePtr out_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));

                return {out_gnode};
            }

            GNodeVector TranslateSlice(const TNodePtr n,
                                       NodeMap& tnode2gnodes,
                                       std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, dim, start, end, step
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                auto dim = GetConstValues<int64>(input_gnodes[1], 1)[0];
                auto start = GetConstValues<int64>(input_gnodes[2], 1)[0];
                auto end = GetConstValues<int64>(input_gnodes[3], 1)[0];
                auto step = GetConstValues<int64>(input_gnodes[4], 1)[0];
                NNFUSION_CHECK(step > 0) << "slice step must greater than 0";

                dim += dim < 0 ? input_shape.size() : 0;
                NNFUSION_CHECK(dim >= 0 && dim < input_shape.size());
                start += start < 0 ? input_shape[dim] : 0;
                end += end < 0 ? input_shape[dim] : 0;

                if (start < 0)
                {
                    start = 0;
                }
                else if (start >= input_shape[dim])
                {
                    start = input_shape[dim];
                }

                if (end < start)
                {
                    end = start;
                }
                else if (end >= input_shape[dim])
                {
                    end = input_shape[dim];
                }

                nnfusion::Coordinate lower(input_shape.size(), 0);
                lower[dim] = start;
                nnfusion::Coordinate upper(input_shape);
                upper[dim] = end;
                nnfusion::Strides strides(input_shape.size(), 1);
                strides[dim] = step;
                auto slice_op = std::make_shared<op::Slice>(lower, upper, strides);
                auto out_gnode = m_graph->add_node_and_edge(slice_op, {input_gnodes[0]});

                return {out_gnode};
            }

            GNodeVector TranslateUnsqueeze(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, dim
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                auto dim = GetConstValues<int64>(input_gnodes[1], 1)[0];
                if (dim < 0)
                {
                    dim += input_shape.size() + 1;
                    NNFUSION_CHECK(dim >= 0 && dim <= input_shape.size());
                }

                auto output_shape = input_shape;
                output_shape.insert(output_shape.begin() + dim, 1);
                nnfusion::AxisVector ng_axis_order(input_shape.size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto reshape_op =
                    std::make_shared<nnfusion::op::Reshape>(ng_axis_order, output_shape);
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_gnodes[0]});

                return {reshape_gnode};
            }

            GNodeVector TranslateTo(const TNodePtr n,
                                    NodeMap& tnode2gnodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs:
                // input, device, dtype, non_blocking, copy
                // input, dtype, non_blocking, copy
                // input, non_blocking, copy
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                c10::ScalarType dtype;
                if (input_gnodes.size() == 3)
                {
                    return {input_gnodes[0]};
                }
                else if (input_gnodes.size() == 4)
                {
                    dtype =
                        static_cast<c10::ScalarType>(GetConstValues<int64>(input_gnodes[1], 1)[0]);
                }
                else if (input_gnodes.size() == 5)
                {
                    dtype =
                        static_cast<c10::ScalarType>(GetConstValues<int64>(input_gnodes[2], 1)[0]);
                }
                else
                {
                    NNFUSION_CHECK_FAIL() << "aten::to accept 3 ~ 5 params, but found"
                                          << input_gnodes.size();
                }

                nnfusion::element::Type ng_et;
                NNFUSION_CHECK(ScalarTypeToNGraphElementType(dtype, &ng_et));

                auto cast_op = std::make_shared<op::Convert>(ng_et);
                auto cast_gnode = m_graph->add_node_and_edge(cast_op, {input_gnodes[0]});

                return {cast_gnode};
            }

            GNodeVector TranslateEmbedding(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: weight, input, padding_idx(int, default -1), scale_grad_by_freq(bool default false), sparse(bool default false)
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);
                auto weight = input_gnodes[0];
                auto input = input_gnodes[1];

                // TODO:  pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.
                // seems it has been done while convert to torchscript, need to be double check
                auto padding_idx = GetConstValues<int64>(input_gnodes[2], 1)[0];

                /*
                if (padding_idx != -1)
                {
                    if (padding_idx < 0)
                    {
                        padding_idx += embedding_shape[0];
                    }
                    NNFUSION_CHECK(padding_idx >= 0 && padding_idx < embedding_shape[0]) << "Padding_idx must be within num_embeddings";
                }
                */

                auto scale_grad_by_freq = GetConstValues<bool>(input_gnodes[3], 1)[0];
                NNFUSION_CHECK(scale_grad_by_freq == 0);
                auto sparse = GetConstValues<bool>(input_gnodes[4], 1)[0];
                NNFUSION_CHECK(sparse == 0);

                nnfusion::op::OpConfig::any myConfig;
                myConfig["axis"] = 0;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    n->kind().toQualString(), "GatherV2", myConfig);

                auto generic_gnode = m_graph->add_node_and_edge(generic_op, {weight, input});
                return {generic_gnode};
            }

            GNodeVector TranslateLayerNorm(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, normalized_shape, weight, bias, eps=1e-05, elementwise_affine=True
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input = input_gnodes[0]; // 1, 20, 768
                auto input_shape = input->get_shape();
                auto input_rank = input_shape.size();

                auto normalized_shape = GetConstValues<int64>(input_gnodes[1]);
                auto normalized_rank = normalized_shape.size();
                auto diff_pair = std::mismatch(input_shape.end() - normalized_shape.size(),
                                               input_shape.end(),
                                               normalized_shape.begin());
                NNFUSION_CHECK(diff_pair.first == input_shape.end() &&
                               diff_pair.second == normalized_shape.end());

                auto weight = input_gnodes[2];
                auto bias = input_gnodes[3];
                auto eps = GetConstValues<float>(input_gnodes[4], 1)[0];
                eps = eps > 0 ? eps : 1e-5;

                auto num_feature = nnfusion::shape_size(normalized_shape);

                // mean
                std::vector<size_t> reduction_axes(normalized_rank);
                std::iota(
                    reduction_axes.begin(), reduction_axes.end(), input_rank - normalized_rank);
                auto sum_gnode = m_graph->add_node_and_edge(
                    std::make_shared<op::Sum>(reduction_axes), {input}); // 1, 20

                const auto& et = sum_gnode->get_element_type();
                auto divisor_op = std::make_shared<op::Constant>(
                    et,
                    sum_gnode->get_shape(),
                    std::vector<std::string>{std::to_string(num_feature)});
                auto divisor_gnode = m_graph->add_node_and_edge(divisor_op, GNodeVector({}));

                auto mean_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                             {sum_gnode, divisor_gnode}); // 1, 20
                // keep dim
                nnfusion::Shape mean_shape_with_keep(input_rank);
                for (size_t i = 0; i < input_rank; i++)
                {
                    mean_shape_with_keep[i] = i < input_rank - normalized_rank ? input_shape[i] : 1;
                }
                nnfusion::AxisVector ng_axis_order(mean_gnode->get_shape().size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                mean_gnode = m_graph->add_node_and_edge(
                    std::make_shared<op::Reshape>(ng_axis_order, mean_shape_with_keep),
                    {mean_gnode}); // 1, 20, 1
                std::tie(input, mean_gnode) =
                    numpy_broadcast(std::make_pair(input, mean_gnode), m_graph); // 1, 20, 768
                mean_gnode = m_graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                                        {input, mean_gnode});

                // std
                auto std_power_gnode = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                                  {mean_gnode, mean_gnode});
                auto std_sum_gnode = m_graph->add_node_and_edge(
                    std::make_shared<op::Sum>(reduction_axes), {std_power_gnode});
                auto std_mean_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                                 {std_sum_gnode, divisor_gnode});
                auto std_sqrt_gnode =
                    m_graph->add_node_and_edge(std::make_shared<op::Sqrt>(), {std_mean_gnode});
                auto eps_op = std::make_shared<op::Constant>(
                    et, std_sqrt_gnode->get_shape(), std::vector<std::string>{std::to_string(eps)});
                auto eps_gnode = m_graph->add_node_and_edge(eps_op, GNodeVector({}));
                auto std_gnode = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                            {std_sqrt_gnode, eps_gnode}); // 1, 20
                // keep dim
                std_gnode = m_graph->add_node_and_edge(
                    std::make_shared<op::Reshape>(ng_axis_order, mean_shape_with_keep),
                    {std_gnode}); // 1, 20, 1
                std::tie(input, std_gnode) =
                    numpy_broadcast(std::make_pair(input, std_gnode), m_graph); // 1, 20, 768

                auto norm_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                             {mean_gnode, std_gnode});

                // weight
                std::tie(input, weight) = numpy_broadcast(std::make_pair(input, weight), m_graph);
                // bias
                std::tie(input, bias) = numpy_broadcast(std::make_pair(input, bias), m_graph);

                auto mul_gnode = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                            {weight, norm_gnode});
                auto ret_gnode =
                    m_graph->add_node_and_edge(std::make_shared<op::Add>(), {mul_gnode, bias});

                return {ret_gnode};
            }

            GNodeVector TranslateRsub(const TNodePtr n,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // aten::rsub accept 3 inputs(y, x, alpha), output = x - alpha * y
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto y = input_gnodes[0];
                auto x = input_gnodes[1];
                auto alpha = input_gnodes[2];

                auto cast_op = std::make_shared<op::Convert>(y->get_element_type());
                alpha = m_graph->add_node_and_edge(cast_op, {alpha});
                std::tie(y, alpha) = numpy_broadcast(std::make_pair(y, alpha), m_graph);

                auto add_right =
                    m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, alpha});

                std::tie(x, add_right) = numpy_broadcast(std::make_pair(x, add_right), m_graph);
                auto out_gnode =
                    m_graph->add_node_and_edge(std::make_shared<op::Subtract>(), {x, add_right});

                return {out_gnode};
            }

            GNodeVector TranslateSelect(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs: input, dim, index
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_shape = input_gnodes[0]->get_shape();
                auto dim = GetConstValues<int64>(input_gnodes[1], 1)[0];
                auto index = GetConstValues<int64>(input_gnodes[2], 1)[0];

                if (dim < 0)
                {
                    dim += input_shape.size();
                    NNFUSION_CHECK(dim >= 0 && dim < input_shape.size());
                }

                nnfusion::Coordinate lower(input_shape.size(), 0);
                lower[dim] = index;
                nnfusion::Coordinate upper(input_shape);
                upper[dim] = index + 1;
                auto slice_op = std::make_shared<op::Slice>(lower, upper);
                auto slice_gnode = m_graph->add_node_and_edge(slice_op, {input_gnodes[0]});

                auto output_shape = input_shape;
                output_shape.erase(output_shape.begin() + dim);
                nnfusion::AxisVector ng_axis_order(input_shape.size());
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                auto reshape_op =
                    std::make_shared<nnfusion::op::Reshape>(ng_axis_order, output_shape);
                auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {slice_gnode});

                return {reshape_gnode};
            }

            GNodeVector TranslateTranspose(const TNodePtr n,
                                           NodeMap& tnode2gnodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto input_dim = input_gnodes[0]->get_shape().size();
                auto dim_1 = GetConstValues<int64>(input_gnodes[1], 1)[0];
                auto dim_2 = GetConstValues<int64>(input_gnodes[2], 1)[0];

                AxisVector ng_axis_order(input_dim);
                std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

                dim_1 += dim_1 < 0 ? input_dim : 0;
                dim_2 += dim_2 < 0 ? input_dim : 0;
                std::swap(ng_axis_order[dim_1], ng_axis_order[dim_2]);

                auto out_gnode = numpy_transpose(input_gnodes[0], ng_axis_order);
                m_graph->add_node(out_gnode);
                m_graph->add_edge(input_gnodes[0], 0, out_gnode, 0);

                return {out_gnode};
            }

            GNodeVector TranslateTupleConstruct(const TNodePtr n,
                                                NodeMap& tnode2gnodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                return {input_gnodes[0]}; // return the 1st element
            }

            GNodeVector TranslateContiguous(const TNodePtr n,
                                            NodeMap& tnode2gnodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // input: tensor, memoryFormat
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                return {input_gnodes[0]};
            }

            GNodeVector TranslateArange(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // input: end, dtype, layout, device, pin_memory
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                // TODO: support more type
                auto dtype = GetConstValues<int64>(input_gnodes[1], 1)[0];
                NNFUSION_CHECK(c10::ScalarType(dtype) == c10::ScalarType::Long);
                auto end = GetConstValues<int64>(input_gnodes[0], 1)[0];

                auto values = std::vector<int64>(end);
                std::iota(values.begin(), values.end(), 0);
                auto out_op = MakeConstOp({static_cast<uint64>(end)}, values);
                auto out_gnode = m_graph->add_node_and_edge(out_op, GNodeVector({}));
                return {out_gnode};
            }

            GNodeVector TranslateLt(const TNodePtr n,
                                    NodeMap& tnode2gnodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector input_gnodes = GetAllInputNode(tnode2gnodes, n);

                auto lhs_gnode = input_gnodes[0];
                auto rhs_gnode = input_gnodes[1];
                if (lhs_gnode->get_element_type() != rhs_gnode->get_element_type())
                {
                    auto cast_op = std::make_shared<op::Convert>(lhs_gnode->get_element_type());
                    rhs_gnode = m_graph->add_node_and_edge(cast_op, {rhs_gnode});
                }
                std::tie(lhs_gnode, rhs_gnode) =
                    numpy_broadcast(std::make_pair(lhs_gnode, rhs_gnode), m_graph);
                auto out_gnode = m_graph->add_node_and_edge(std::make_shared<op::Less>(),
                                                            {lhs_gnode, rhs_gnode});
                return {out_gnode};
            }

            GNodeVector TranslateExpandAs(const TNodePtr n,
                                          NodeMap& tnode2gnodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // input: input, tensor with expected shape
                auto input = GetInputNode(tnode2gnodes, n, 0);
                auto expect = GetInputNode(tnode2gnodes, n, 1);

                std::tie(input, expect) = numpy_broadcast(std::make_pair(input, expect), m_graph);

                return {input};
            }

            GNodeVector TranslateMaskedFill(const TNodePtr n,
                                            NodeMap& tnode2gnodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // input: input, mask, filled_value
                auto input = GetInputNode(tnode2gnodes, n, 0);
                auto mask = GetInputNode(tnode2gnodes, n, 1);
                auto value = GetInputNode(tnode2gnodes, n, 2);
                if (mask->get_element_type() != nnfusion::element::boolean)
                {
                    mask = m_graph->add_node_and_edge(
                        std::make_shared<op::Convert>(nnfusion::element::boolean), {mask});
                }
                std::tie(input, mask) = numpy_broadcast(std::make_pair(input, mask), m_graph);
                value = m_graph->add_node_and_edge(
                    std::make_shared<op::Convert>(input->get_element_type()), {value});
                std::tie(input, value) = numpy_broadcast(std::make_pair(input, value), m_graph);

                auto out = m_graph->add_node_and_edge(std::make_shared<op::Select>(),
                                                      {mask, value, input});
                return {out};
            }

            GNodeVector TranslateTypeAs(const TNodePtr n,
                                        NodeMap& tnode2gnodes,
                                        std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // input: input, expect_type_tensor
                auto input = GetInputNode(tnode2gnodes, n, 0);
                auto expect = GetInputNode(tnode2gnodes, n, 1);

                auto out = m_graph->add_node_and_edge(
                    std::make_shared<op::Convert>(expect->get_element_type()), {input});
                return {out};
            }

            const static std::map<const std::string, ConvertFunc> TRANSLATE_OP_MAP{
                {"aten::add", TranslateAdd},
                {"aten::add_", TranslateAdd},
                {"aten::mul", TranslateBinaryOp<op::Multiply>},
                {"aten::mul_", TranslateBinaryOp<op::Multiply>},
                {"prim::Constant", TranslateConstant},
                {"aten::mv", TranslateDot},
                {"aten::dim", TranslateDim},
                {"aten::eq", TranslateEq},
                {"prim::Return", TranslateReturn},
                {"prim::If", TranslateIf},
                {"aten::t", TranslateT},
                {"aten::matmul", TranslateMatMul},
                {"aten::__isnot__", Translate__isnot__},
                {"prim::unchecked_unwrap_optional", TranslateNo},
                {"prim::ListConstruct", TranslateListConstruct},
                {"aten::_convolution", TranslateConvolution2D},
                {"aten::relu_", TranslateUnaryOp<op::Relu>},
                {"aten::max_pool2d", TranslateMaxPool2D},
                {"aten::adaptive_avg_pool2d", TranslateAdaptiveAvgPool2d},
                {"aten::size", TranslateSize},
                {"prim::NumToTensor", TranslateNo},
                {"aten::Int", TranslateInt},
                {"aten::view", TranslateView},
                {"aten::flatten", TranslateFlatten},
                {"aten::addmm", TranslateAddmm},
                {"aten::dropout", TranslateDropout},
                {"aten::tanh", TranslateUnaryOp<op::Tanh>},
                {"aten::permute", TranslatePermute},
                {"aten::expand", TranslateExpand},
                {"aten::softmax", TranslateSoftMax},
                {"aten::div", TranslateBinaryOp<op::Divide>},
                {"aten::device", TranslateNone},
                {"aten::zeros", TranslateZeros},
                {"aten::ones", TranslateOnes},
                {"aten::slice", TranslateSlice},
                {"aten::unsqueeze", TranslateUnsqueeze},
                {"aten::to", TranslateTo},
                {"aten::embedding", TranslateEmbedding},
                {"aten::layer_norm", TranslateLayerNorm},
                {"aten::rsub", TranslateRsub},
                {"prim::ImplicitTensorToNum", TranslateNo},
                {"prim::TupleConstruct",
                 TranslateTupleConstruct}, // tuple used for multiple returns
                {"aten::select", TranslateSelect},
                {"aten::transpose", TranslateTranspose},
                {"aten::contiguous", TranslateContiguous},
                {"aten::arange", TranslateArange},
                {"aten::erf", TranslateUnaryOp<op::Erf>},
                {"aten::lt", TranslateLt},
                {"aten::expand_as", TranslateExpandAs},
                {"aten::masked_fill_", TranslateMaskedFill},
                {"aten::type_as", TranslateTypeAs},
                {"aten::detach", TranslateNo}};

            GNodeVector TranslateParam(TNodePtr n,
                                       std::shared_ptr<nnfusion::graph::Graph> m_graph,
                                       NodeMap tnode2gnodes,
                                       std::vector<Shape> shapes,
                                       std::vector<element::Type> types,
                                       std::vector<at::Tensor> weights)
            {
                GNodeVector ret;
                auto params = n->outputs();
                NNFUSION_CHECK(shapes.size() == types.size()) << "Input shape doesn\'t match types";
                NNFUSION_CHECK(shapes.size() + weights.size() == params.size())
                    << "No enough static info for graph input";
                auto num_inputs = shapes.size();
                auto num_weights = weights.size();
                // load actual model inputs as parameter
                for (auto i = 0; i < num_inputs; i++)
                {
                    auto para_op = std::make_shared<op::Parameter>(types[i], shapes[i]);
                    para_op->set_name(params[i]->debugName());

                    auto para_gnode = m_graph->add_node_and_edge(para_op, GNodeVector({}));
                    ret.push_back(para_gnode);
                }

                // load weights as const
                for (auto i = 0; i < num_weights; i++)
                {
                    auto const_op = MakeConstOp(weights[i]);
                    const_op->set_name(params[num_inputs + i]->debugName());

                    auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));
                    ret.push_back(const_gnode);
                }
                return ret;
            }

            void print_gnode_input_output(GNodePtr n)
            {
                int arg_cnt = 0;
                int depth = 0;
                auto in_edges_set = n->get_in_edges();
                std::vector<std::shared_ptr<nnfusion::graph::Edge>> in_edges(in_edges_set.begin(),
                                                                             in_edges_set.end());
                std::sort(in_edges.begin(),
                          in_edges.end(),
                          [](std::shared_ptr<nnfusion::graph::Edge> a,
                             std::shared_ptr<nnfusion::graph::Edge> b) {
                              return a->get_dst_input() < b->get_dst_input();
                          });

                std::cout << "Inputs: " << std::endl;
                for (auto in_edge : in_edges)
                {
                    std::cout << "input_index: " << arg_cnt++ << ",";
                    auto input_node = in_edge->get_src();
                    std::cout << "shape: " << input_node->get_shape() << ", ";
                    std::vector<double> output;
                    GetValueFromNGraphOp<double>(input_node, &output);
                    std::cout << "elements: ";
                    for (int i = 0; i < std::min(10LU, output.size()); ++i)
                        std::cout << output[i] << ", ";
                    std::cout << std::endl;
                }

                std::cout << "Outputs: " << std::endl;
                std::vector<double> output;
                std::cout << "shape: " << n->get_shape() << ", ";
                GetValueFromNGraphOp<double>(n, &output);
                std::cout << "elements: ";
                for (int i = 0; i < std::min(10LU, output.size()); ++i)
                    std::cout << output[i] << ", ";
                std::cout << std::endl;
            }

            GNodeVector convert_node(const TNodePtr node,
                                     NodeMap& tnode2gnodes,
                                     std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                GNodeVector ret;
#if TS_IMPORT_DEBUG // for Debug
                if (strcmp(node->kind().toQualString(), "prim::Return") != 0)
                {
                    std::cout << "converting: " << node->output()->debugName() << ", "
                              << node->kind().toQualString() << std::endl;
                }
#endif
                auto func = TRANSLATE_OP_MAP.find(node->kind().toQualString());
                if (func != TRANSLATE_OP_MAP.end())
                {
                    ret = func->second(node, tnode2gnodes, m_graph);
#if TS_IMPORT_DEBUG // for Debug
                    if (ret[0] && strcmp(node->kind().toQualString(), "prim::Return") != 0)
                    {
                        std::unordered_set<string> debug_nodes{};
                        if (string(node->output()->debugName()) == "1185")
                        {
                            std::cout << 1 << std::endl;
                        }
                        if (debug_nodes.find(node->output()->debugName()) != debug_nodes.end())
                        {
                            std::cout << "node name: " << string(node->output()->debugName())
                                      << std::endl;
                            print_gnode_input_output(ret[0]);
                        }

                        std::cout << "output shape: " << ret[0]->get_shape() << ", output type "
                                  << ret[0]->get_element_type() << std::endl;
                    }
#endif
                }
                else
                {
                    // TODO: convertFunc not found
                    NNFUSION_CHECK_FAIL() << "Convert func for " << node->kind().toQualString()
                                          << " not found";
                }
                return std::move(ret);
            }

            GNodeVector convert_block(const TBlockPtr block,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                return convert_block(block,
                                     std::vector<at::Tensor>(),
                                     std::vector<nnfusion::Shape>(),
                                     std::vector<nnfusion::element::Type>(),
                                     tnode2gnodes,
                                     m_graph);
            }

            GNodeVector convert_block(const TBlockPtr block,
                                      const std::vector<at::Tensor>& weights,
                                      const std::vector<nnfusion::Shape>& input_shapes,
                                      const std::vector<nnfusion::element::Type>& input_types,
                                      NodeMap& tnode2gnodes,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                // inputs
                NNFUSION_CHECK(input_shapes.size() == input_types.size())
                    << "Input shapes and types size do not match.";
                auto input_gnodes = TranslateParam(
                    block->param_node(), m_graph, tnode2gnodes, input_shapes, input_types, weights);
                tnode2gnodes[block->param_node()] = input_gnodes;
#if TS_IMPORT_DEBUG // for Debug
                for (auto n : input_gnodes)
                {
                    std::cout << "converting input: " << n->get_name() << "," << n->get_op_type()
                              << std::endl;
                    std::cout << "output shape: " << n->get_shape() << std::endl;
                }
#endif

                // common ops
                for (auto n : block->nodes())
                {
                    auto gnodes = convert_node(n, tnode2gnodes, m_graph);
                    tnode2gnodes[n] = gnodes;
                }

                // outputs
                auto return_gnodes = convert_node(block->return_node(), tnode2gnodes, m_graph);
                return return_gnodes;
            }

            void print_node_inputs(TNodePtr n, int indent = 0)
            {
                std::cout << std::string(indent, '\t') << "(" << n << ")" << *n;
                c10::ArrayRef<TValuePtr> inputs = n->inputs();
                for (int i = 0; i < inputs.size(); i++)
                {
                    TValuePtr v = inputs[i];
                    std::cout << std::string(indent, '\t') << "value: " << v->debugName() << ", "
                              << *v->type() << std::endl;
                    std::cout << std::string(indent, '\t')
                              << "edge: " << v->node()->kind().toQualString() << "(" << v->node()
                              << "): " << v->offset() << " -> " << n->kind().toQualString() << ": "
                              << i << std::endl;
                }
            }

            void print_block(torch::jit::Block* b, int indent = 0)
            {
                // inputs
                std::cout << std::string(indent, '\t') << "Param node:" << std::endl;
                TNodePtr param = b->param_node();
                print_node_inputs(param, indent);
                std::cout << std::endl;

                // ops
                std::cout << std::string(indent, '\t') << "Op node:" << std::endl;
                torch::jit::graph_node_list nodes = b->nodes();
                for (const auto& n : nodes)
                {
                    print_node_inputs(n, indent);
                    auto attrs = n->attributeNames();

                    for (int i = 0; i < n->blocks().size(); i++)
                    {
                        std::cout << std::string(indent, '\t') << "subblock: " << i << std::endl;
                        auto block = n->blocks()[i];
                        print_block(block, indent + 1);
                    }
                    std::cout << std::endl;
                }

                // outputs
                std::cout << std::string(indent, '\t') << "Return node:" << std::endl;
                TNodePtr return_node = b->return_node();
                print_node_inputs(return_node, indent);
                std::cout << std::endl;
            }

            GraphConvert::GraphConvert(const std::shared_ptr<torch::jit::Graph> ts_graph,
                                       const std::vector<at::Tensor>& weights,
                                       const std::vector<nnfusion::Shape>& input_shapes,
                                       const std::vector<nnfusion::element::Type>& input_types)
                : ts_graph_{ts_graph}
                , weights_{weights}
                , input_shapes_{input_shapes}
                , input_types_{input_types}
                , m_graph(new nnfusion::graph::Graph())

            {
#if TS_IMPORT_DEBUG // for Debug
                std::cout << *ts_graph_ << std::endl;
#endif

                auto return_gnodes = convert_block(ts_graph_->block(),
                                                   weights_,
                                                   input_shapes_,
                                                   input_types_,
                                                   tnode2gnodes,
                                                   m_graph);

                m_graph->set_outputs(return_gnodes);

                NNFUSION_LOG(INFO) << "Convert Torchscript Graph Done";
            }
        } // namespace torchscript_import
    }     // namespace frontend
} // namespace nnfusion
