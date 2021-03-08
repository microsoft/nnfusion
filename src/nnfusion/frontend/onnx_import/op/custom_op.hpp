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

#include "core/attribute.hpp"
#include "core/node.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateCustomOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    Node node(node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    std::vector<Attribute> attrs{std::begin(node_proto.attribute()),
                                                 std::end(node_proto.attribute())};
                    for (const auto& attr : attrs)
                    {
                        if (attr.is_float())
                        {
                            myConfig[attr.get_name()] = attr.get_float();
                        }
                        else if (attr.is_float_array())
                        {
                            myConfig[attr.get_name()] = attr.get_float_array();
                        }
                        else if (attr.is_integer())
                        {
                            myConfig[attr.get_name()] = attr.get_integer();
                        }
                        else if (attr.is_integer_array())
                        {
                            myConfig[attr.get_name()] = attr.get_integer_array();
                        }
                        else if (attr.is_string())
                        {
                            myConfig[attr.get_name()] = attr.get_string();
                        }
                        else if (attr.is_string_array())
                        {
                            myConfig[attr.get_name()] = attr.get_string_array();
                        }
                        else if (attr.is_tensor())
                        {
                            // Tensor attribute will be flatten
                            Tensor t = attr.get_tensor();
                            auto type = t.get_ng_type();
                            if (type == element::f32)
                            {
                                myConfig[attr.get_name()] = t.get_data<float>();
                            }
                            else if (type == element::f64)
                            {
                                myConfig[attr.get_name()] = t.get_data<double>();
                            }
                            else if (type == element::i32)
                            {
                                myConfig[attr.get_name()] = t.get_data<int32_t>();
                            }
                            else if (type == element::i64)
                            {
                                myConfig[attr.get_name()] = t.get_data<int64_t>();
                            }
                            else if (type == element::u32)
                            {
                                myConfig[attr.get_name()] = t.get_data<uint32_t>();
                            }
                            else if (type == element::u64)
                            {
                                myConfig[attr.get_name()] = t.get_data<uint64_t>();
                            }
                            else
                            {
                                NNFUSION_CHECK_FAIL() << "Unsupported attribute type: Tensor "
                                                      << type;
                            }
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL();
                        }
                    }

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), node_proto.op_type(), myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(
                        generic_op, input_indexes, node_proto.output_size());

                    NamedNodeVector ret;
                    for (size_t i = 0; i < node_proto.output_size(); i++)
                    {
                        ret.emplace_back(node_proto.output(i), generic_gnode, i);
                    }

                    return ret;
                }
            } // namespace set_1
            const static ConvertFunc custom_translator = set_1::TranslateCustomOp;
        } //namespace onnx_import
    }     // namespace frontend
} // namespace  nnfusion
