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

#pragma once

#include "core/node.hpp"
#include "core/tensor.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_9
            {
                NamedNodeVector
                    TranslateConstantOfShapeOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto input = input_indexes[0];
                    std::vector<int64> output_shape;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input.gnode, &output_shape));

                    Node node(node_proto);
                    std::shared_ptr<nnfusion::op::Constant> const_op;
                    NNFUSION_LOG(INFO) << Shape(std::begin(output_shape), std::end(output_shape));
                    if (node.has_attribute("value"))
                    {
                        auto value = node.get_attribute_value<Tensor>("value");
                        NNFUSION_CHECK(nnfusion::shape_size(value.get_shape()) == 1);
                        const_op = make_constant_op(
                            value.get_ng_type(),
                            Shape(std::begin(output_shape), std::end(output_shape)),
                            value);
                    }
                    else
                    {
                        auto vec = std::vector<float>{0};
                        const_op = std::make_shared<op::Constant>(element::f32, Shape{1}, vec);
                    }

                    const_op->set_name(node_proto.output(0));
                    const_op->set_global_consistent_name(node_proto.output(0));
                    auto const_gnode = m_graph->add_node_and_edge(const_op, graph::GNodeVector({}));

                    return {{node_proto.output(0), const_gnode}};
                }

            } // namespace set_9
        }     //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
