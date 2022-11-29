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

#include "util.hpp"
#include "../core/node.hpp"
#include "../core/tensor.hpp"
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            const nnfusion::element::Type& ONNXDataTypeToNNFusionElementType(int onnx_dt)
            {
                switch (onnx_dt)
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL: return element::boolean;
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: return element::f32;
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16: return element::f16;
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: return element::f64;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8: return element::i8;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16: return element::i16;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32: return element::i32;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64: return element::i64;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8: return element::u8;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16: return element::u16;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32: return element::u32;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64: return element::u64;
                default:
                    NNFUSION_CHECK_FAIL() << "unsupported onnx element type: "
                                          << detail::onnx_type_name(onnx_dt);
                }
                return element::f32;
            }

            std::shared_ptr<op::Constant> make_constant_op(const element::Type& type,
                                                           const Shape& shape,
                                                           const Tensor& tensor)
            {
                if (type == element::boolean)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<int16_t>());
                if (type == element::i8)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<int8_t>());
                if (type == element::i16)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<int16_t>());
                if (type == element::i32)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<int32_t>());
                if (type == element::i64)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<int64_t>());
                if (type == element::u8)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<uint8_t>());
                if (type == element::u16)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<uint16_t>());
                if (type == element::u32)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<uint32_t>());
                if (type == element::u64)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<uint64_t>());
                if (type == element::f16)
                    return std::make_shared<op::Constant>(
                        type, shape, tensor.get_data<half_float::half>());
                if (type == element::f32)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<float>());
                if (type == element::f64)
                    return std::make_shared<op::Constant>(type, shape, tensor.get_data<double>());
                NNFUSION_CHECK_FAIL() << "unsupported element type: " << type;
                return std::make_shared<op::Constant>(type, shape, tensor.get_data<float>());
            }

            std::shared_ptr<op::Constant> make_constant_op(const Tensor& tensor)
            {
                return make_constant_op(tensor.get_ng_type(), tensor.get_shape(), tensor);
            }

            std::shared_ptr<graph::GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                       const onnx::NodeProto& node,
                                                       size_t input_idx)
            {
                std::shared_ptr<graph::GNode> result = nullptr;
                try
                {
                    result = all_ng_nodes.at(node.input(input_idx)).at(0).gnode;
                }
                catch (const std::out_of_range&)
                {
                    if (node.input(input_idx) == "")
                    {
                        return nullptr;
                    }
                    NNFUSION_CHECK_FAIL() << "Input op not found for " << node.input(input_idx);
                }
                return result;
            }

            // TODO: replace all GetInputNode to GetInputIndex
            GNodeIndex GetInputIndex(const NodeMap& all_ng_nodes,
                                     const onnx::NodeProto& node,
                                     size_t input_idx)
            {
                GNodeIndex result{nullptr};
                try
                {
                    result = all_ng_nodes.at(node.input(input_idx)).at(0);
                }
                catch (const std::out_of_range&)
                {
                    if (node.input(input_idx) == "")
                    {
                        return GNodeIndex{nullptr};
                    }
                    NNFUSION_CHECK_FAIL() << "Input op not found for " << node.input(input_idx);
                }
                return result;
            }

            graph::GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                               const onnx::NodeProto& node)
            {
                graph::GNodeVector nodes;
                for (size_t i = 0; i < node.input_size(); i++)
                {
                    nodes.push_back(GetInputNode(all_ng_nodes, node, i));
                }
                return nodes;
            }

            GNodeIndexVector GetAllInputIndex(const NodeMap& all_ng_nodes,
                                              const onnx::NodeProto& node)
            {
                GNodeIndexVector indexes;
                for (size_t i = 0; i < node.input_size(); i++)
                {
                    indexes.push_back(GetInputIndex(all_ng_nodes, node, i));
                }
                return indexes;
            }

            Shape get_kernel_shape(const Node& node,
                                   const std::shared_ptr<graph::GNode> input_gnode)
            {
                std::size_t input_spacial_dims = input_gnode->get_shape().size() - 2;
                return node.get_attribute_value<std::vector<std::size_t>>(
                    "kernel_shape", std::vector<std::size_t>(input_spacial_dims, 1UL));
            }

            Strides get_strides(const Node& node, const Shape& kernel_shape)
            {
                return get_strides_helper(node, "strides", kernel_shape);
            }

            Strides get_strides(const Node& node, const std::shared_ptr<graph::GNode> input_gnode)
            {
                return get_strides(node, get_kernel_shape(node, input_gnode));
            }

            Strides get_dilations(const Node& node, const std::shared_ptr<graph::GNode> input_gnode)
            {
                return get_strides_helper(node, "dilations", get_kernel_shape(node, input_gnode));
            }

            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape)
            {
                CoordinateDiff pads;
                if (node.has_attribute("pads"))
                {
                    auto pads_int64 = node.get_attribute_value<std::vector<int64_t>>("pads");
                    pads = CoordinateDiff{std::begin(pads_int64), std::end(pads_int64)};
                }
                else
                {
                    std::string auto_pad{node.get_attribute_value<std::string>("auto_pad", "")};
                    if (!auto_pad.empty())
                    {
                        pads = get_auto_pads(kernel_shape, auto_pad);
                    }
                }
                if (pads.empty())
                {
                    pads = CoordinateDiff(static_cast<std::ptrdiff_t>(kernel_shape.size()), 0UL);
                }

                if (pads.size() != kernel_shape.size() * 2)
                {
                    // Paddings specified in (H, W, C) format.
                    return {pads, pads};
                }
                else
                {
                    return {{std::begin(pads) + pads.size() / 2, std::end(pads)},
                            {std::begin(pads), std::begin(pads) + pads.size() / 2}};
                }
            }

            CoordinateDiff get_auto_pads(const Shape& kernel_shape, const std::string& auto_pad)
            {
                CoordinateDiff pads;

                // Add padding to the input to match the size of output size.
                auto pad_value = [](size_t dim) { return (static_cast<float>(dim) - 1.f) / 2.f; };

                if (auto_pad == "SAME_UPPER")
                {
                    for (size_t dim : kernel_shape)
                    {
                        pads.emplace_back(std::floor(pad_value(dim)));
                    }
                    for (size_t dim : kernel_shape)
                    {
                        pads.emplace_back(std::ceil(pad_value(dim)));
                    }
                }
                else if (auto_pad == "SAME_LOWER")
                {
                    for (size_t dim : kernel_shape)
                    {
                        pads.emplace_back(std::ceil(pad_value(dim)));
                    }
                    for (size_t dim : kernel_shape)
                    {
                        pads.emplace_back(std::floor(pad_value(dim)));
                    }
                }

                return pads;
            }

            Strides get_strides_helper(const Node& node,
                                       const std::string& name,
                                       const Shape& kernel_shape)
            {
                return node.get_attribute_value<std::vector<std::size_t>>(
                    name, std::vector<std::size_t>(kernel_shape.size(), 1UL));
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
