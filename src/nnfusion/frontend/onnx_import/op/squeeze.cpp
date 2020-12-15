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

//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include "squeeze.hpp"
#include "util/reshape.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSqueezeOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto data_shape = data.get_shape();

                    Node node(node_proto);
                    auto axes = node.get_attribute_value<std::vector<std::size_t>>("axes", {});
                    AxisVector input_order{reshape::get_default_axis_vector(data_shape.size())};

                    // Prepare set of unique axes marked to be removed from input data.
                    if (axes.empty())
                    {
                        // Default behaviour is to remove all single dimension axes.
                        for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                        {
                            if (data_shape.at(idx) == 1)
                            {
                                // Mark with zero elements to remove;
                                data_shape.at(idx) = 0;
                            }
                        }
                    }
                    else
                    {
                        std::set<std::size_t, std::greater<std::size_t>> unique_axes(
                            std::begin(axes), std::end(axes));
                        for (uint64_t axis : unique_axes)
                        {
                            NNFUSION_CHECK(data_shape.at(axis) == 1)
                                << "provided axis value is invalid. Only single dimension axes may "
                                   "be removed.";
                            // Mark with zero elements to remove;
                            data_shape.at(axis) = 0;
                        }
                    }

                    Shape output_data_shape;
                    for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                    {
                        if (data_shape.at(idx) != 0)
                        {
                            output_data_shape.push_back(data_shape.at(idx));
                        }
                    }

                    auto reshape_op = std::make_shared<op::Reshape>(input_order, output_data_shape);
                    reshape_op->set_name(node_proto.output(0));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {data});

                    return {{node_proto.output(0), reshape_gnode}};
                }
            } // namespace set_1

            namespace set_11
            {
                NamedNodeVector TranslateSqueezeOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto data_shape = data.get_shape();

                    Node node(node_proto);
                    auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {});
                    for (auto& axis : axes)
                    {
                        axis += axis < 0 ? data_shape.size() : 0;
                    }
                    AxisVector input_order{reshape::get_default_axis_vector(data_shape.size())};

                    // Prepare set of unique axes marked to be removed from input data.
                    if (axes.empty())
                    {
                        // Default behaviour is to remove all single dimension axes.
                        for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                        {
                            if (data_shape.at(idx) == 1)
                            {
                                // Mark with zero elements to remove;
                                data_shape.at(idx) = 0;
                            }
                        }
                    }
                    else
                    {
                        std::set<std::int64_t, std::greater<std::int64_t>> unique_axes(
                            std::begin(axes), std::end(axes));
                        for (int64_t axis : unique_axes)
                        {
                            NNFUSION_CHECK(data_shape.at(axis) == 1)
                                << "provided axis value is invalid. Only single dimension axes may "
                                   "be removed.";
                            // Mark with zero elements to remove;
                            data_shape.at(axis) = 0;
                        }
                    }

                    Shape output_data_shape;
                    for (std::size_t idx = 0; idx < data_shape.size(); ++idx)
                    {
                        if (data_shape.at(idx) != 0)
                        {
                            output_data_shape.push_back(data_shape.at(idx));
                        }
                    }

                    auto reshape_op = std::make_shared<op::Reshape>(input_order, output_data_shape);
                    reshape_op->set_name(node_proto.output(0));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {data});

                    return {{node_proto.output(0), reshape_gnode}};
                }
            } // namespace set_11

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
