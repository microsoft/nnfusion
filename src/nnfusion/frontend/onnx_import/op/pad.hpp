//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslatePadOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    /*
                     * since opset 11, 'pads' and 'value' have been moved from attributes to inputs
                     * */
                    if (node_proto.attribute_size() == 1)
                    {
                        auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                        auto padding_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                        std::vector<int64> paddings;
                        bool status = GetValueFromNGraphOp<int64>(padding_gnode, &paddings);
                        NNFUSION_CHECK(status);

                        NNFUSION_CHECK(paddings.size() % 2 == 0) << "Constant node for paddings "
                                                                    "does not have an even number "
                                                                    "of elements";

                        nnfusion::Shape padding_below(paddings.size() / 2);
                        nnfusion::Shape padding_above(paddings.size() / 2);
                        nnfusion::Shape padding_interior(paddings.size() / 2);

                        for (size_t i = 0; i < paddings.size() / 2; i++)
                        {
                            padding_below[i] = paddings[i];
                            padding_above[i] = paddings[i + paddings.size() / 2];
                            padding_interior[i] = 0;
                        }

                        auto pad_val_op =
                            std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                           nnfusion::Shape{},
                                                           std::vector<std::string>{"0"});
                        auto pad_val_gnode =
                            m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));

                        auto pad_op = std::make_shared<op::Pad>(
                            padding_below, padding_above, padding_interior);
                        pad_op->set_name(node_proto.output(0));

                        auto pad_gnode =
                            m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});

                        NamedNodeVector ret{{node_proto.output(0), pad_gnode}};
                        return ret;
                    }
                    else
                    {
                        cout << "meet pad op" << endl;
                        /* for pad op, 0: mode, 1: pads, 2: constant
                         * we can use attr.name() to get the name of the attr
                         * for mode, attr.s() represents name
                         * for pads, attr.ints(i) get's padding value
                         * for value, attr.f() get's value
                         */
                        auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                        const onnx::AttributeProto& modeAttr = node_proto.attribute(0);
                        cout << modeAttr.name() << endl;
                        if (modeAttr.s() != "constant")
                            NNFUSION_CHECK_FAIL() << "unsupported padding type: " << modeAttr.s();
                        const onnx::AttributeProto& padAttr = node_proto.attribute(1);
                        cout << padAttr.name() << endl;
                        for (int i = 0; i < 8; ++i)
                        {
                            cout << padAttr.ints(i) << ' ';
                        }
                        cout << endl;
                        const onnx::AttributeProto& valueAttr = node_proto.attribute(2);
                        cout << valueAttr.name() << endl;
                        cout << valueAttr.f() << endl;
                        auto pad_val_op =
                            std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                           nnfusion::Shape{},
                                                           std::vector<std::string>{"0"});
                        auto pad_val_gnode =
                            m_graph->add_node_and_edge(pad_val_op, GNodeVector({}));
                        nnfusion::Shape padding_below(4);
                        nnfusion::Shape padding_above(4);
                        nnfusion::Shape padding_interior(4);
                        /*
                         * fix: the correspondence of padding to pads is wrong
                         * */
                        for (int i = 0; i < 4; ++i)
                        {
                            padding_below[i] = padAttr.ints(i);
                            padding_above[i] = padAttr.ints(i + 4);
                        }

                        auto pad_op = std::make_shared<op::Pad>(
                            padding_below, padding_above, padding_interior);
                        pad_op->set_name(node_proto.output(0));

                        auto pad_gnode =
                            m_graph->add_node_and_edge(pad_op, {input_gnode, pad_val_gnode});

                        NamedNodeVector ret{{node_proto.output(0), pad_gnode}};
                        return ret;
                    }
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
