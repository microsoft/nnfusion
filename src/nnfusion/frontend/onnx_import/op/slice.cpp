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

#include <vector>

#include "../util/util.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"
#include "slice.hpp"

static inline int64_t get_valid_array_idx(int64_t idx, int64_t last_idx)
{
    return (idx >= 0) ? std::min(idx, last_idx) : std::max<int64_t>(0, last_idx + idx);
}

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSliceOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    Shape data_shape = data.get_shape();

                    Node node(node_proto);
                    auto starts = node.get_attribute_value<std::vector<int64_t>>("starts");
                    auto ends = node.get_attribute_value<std::vector<int64_t>>("ends");

                    auto axes = node.get_attribute_value<std::vector<int64_t>>(
                        "axes", get_monotonic_range<int64_t>(data_shape.size()));

                    Shape lower_bounds(data_shape.size());
                    Shape upper_bounds = data_shape;

                    for (auto idx = 0; idx < axes.size(); ++idx)
                    {
                        size_t axis = axes.at(idx);
                        lower_bounds.at(axis) =
                            get_valid_array_idx(starts.at(idx), data_shape.at(axis));
                        upper_bounds.at(axis) =
                            get_valid_array_idx(ends.at(idx), data_shape.at(axis));
                    }

                    auto op = std::make_shared<op::Slice>(lower_bounds, upper_bounds);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {data});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1

            namespace set_10
            {
                NamedNodeVector TranslateSliceOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto inputs = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto data = inputs[0];
                    std::vector<int64_t> starts;
                    NNFUSION_CHECK(GetValueFromNGraphOp(inputs[1].gnode, &starts));
                    std::vector<int64_t> ends;
                    NNFUSION_CHECK(GetValueFromNGraphOp(inputs[2].gnode, &ends));
                    std::vector<int64_t> axes;
                    if (inputs.size() > 3)
                    {
                        NNFUSION_CHECK(GetValueFromNGraphOp(inputs[3].gnode, &axes));
                    }
                    else
                    {
                        axes.resize(starts.size());
                        std::iota(axes.begin(), axes.end(), 0);
                    }

                    std::vector<int64_t> steps;
                    if (inputs.size() > 4)
                    {
                        NNFUSION_CHECK(GetValueFromNGraphOp(inputs[4].gnode, &steps));
                    }
                    else
                    {
                        steps.resize(starts.size(), 1);
                    }

                    Shape data_shape = data.get_shape();
                    Shape lower_bounds(data_shape.size());
                    Shape upper_bounds = data_shape;
                    Strides strides(data_shape.size(), 1);

                    for (auto idx = 0; idx < axes.size(); ++idx)
                    {
                        size_t axis = axes.at(idx);
                        lower_bounds.at(axis) =
                            get_valid_array_idx(starts.at(idx), data_shape.at(axis));
                        upper_bounds.at(axis) =
                            get_valid_array_idx(ends.at(idx), data_shape.at(axis));
                        strides.at(axis) = steps.at(idx);
                    }

                    auto op = std::make_shared<op::Slice>(lower_bounds, upper_bounds, strides);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {data});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_10

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
