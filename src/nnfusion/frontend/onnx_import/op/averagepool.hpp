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
                NamedNodeVector
                    TranslateAveragePoolOp(const onnx::NodeProto& node_proto,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    std::vector<int64_t> i_tf_strides;
                    std::vector<int64_t> i_ng_kernel_shape(2);

                    Node node(node_proto);

                    i_tf_strides = node.get_attribute_value<std::vector<std::int64_t>>("strides");
                    i_ng_kernel_shape =
                        node.get_attribute_value<std::vector<std::int64_t>>("kernel_shape");

                    nnfusion::Shape ng_kernel_shape(i_ng_kernel_shape.begin(),
                                                    i_ng_kernel_shape.end());
                    nnfusion::Shape ng_strides(i_tf_strides.begin(), i_tf_strides.end());
                    auto ng_image_shape = input_gnode->get_shape();

                    nnfusion::Shape ng_padding_below{0, 0};
                    nnfusion::Shape ng_padding_above{0, 0};

                    auto avgpool_op = std::make_shared<op::AvgPool>(
                        ng_kernel_shape, ng_strides, ng_padding_below, ng_padding_above, false);
                    auto avgpool_gnode = m_graph->add_node_and_edge(avgpool_op, {input_gnode});
                    avgpool_gnode->set_name(node_proto.output(0));

                    NamedNodeVector ret{{node_proto.output(0), avgpool_gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion