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

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                template <typename T>
                std::shared_ptr<op::Constant> __make_constant_op(const element::Type& type,
                                                                 const Tensor& tensor)
                {
                    return std::make_shared<op::Constant>(
                        type, tensor.get_shape(), tensor.get_data<T>());
                }

                const std::map<element::Type,
                               std::function<std::shared_ptr<op::Constant>(const element::Type&,
                                                                           const Tensor&)>>&
                    ONNX_CONST_MAP()
                {
                    static const std::map<element::Type,
                                          std::function<std::shared_ptr<op::Constant>(
                                              const element::Type&, const Tensor&)>>
                        the_map = {{element::f32, __make_constant_op<float>},
                                   {element::f64, __make_constant_op<double>},
                                   {element::i32, __make_constant_op<int32_t>},
                                   {element::i64, __make_constant_op<int64_t>},
                                   {element::u32, __make_constant_op<uint32_t>},
                                   {element::u64, __make_constant_op<uint64_t>}};

                    return the_map;
                }

                NamedNodeVector TranslateConstantOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    Node node(node_proto);
                    auto tensor = node.get_attribute_value<Tensor>("value");

                    const auto& func_param = ONNX_CONST_MAP().at(tensor.get_ng_type());
                    auto op = func_param(tensor.get_ng_type(), tensor);

                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, graph::GNodeVector({}));
                    NamedNodeVector ret{{node_proto.output(0), gnode}};

                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
