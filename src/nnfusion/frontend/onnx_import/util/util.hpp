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

#include <cmath>       // std::floor
#include <cstddef>     // std::size_t
#include <iterator>    // std::begin, std::end
#include <type_traits> // std::enable_if, std::is_floating_point, std::is_integral
#include <vector>

#include "../onnx_base.hpp"
#include "nnfusion/common/common.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace detail
            {
                template <typename T, typename Container>
                inline std::vector<T> __get_data(const Container& container)
                {
                    return {std::begin(container), std::end(container)};
                }

                template <typename T>
                inline std::vector<T> __get_raw_data(const std::string& raw_data)
                {
                    auto it = reinterpret_cast<const T*>(raw_data.data());
                    return {it, it + (raw_data.size() / sizeof(T))};
                }

                template <typename T>
                inline std::vector<T> get_data(const onnx::TensorProto& tensor)
                {
                    NNFUSION_CHECK_FAIL()
                        << "unsupported data type: "
                        << static_cast<onnx::TensorProto_DataType>(tensor.data_type());
                }

                template <>
                inline std::vector<double> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<double>(tensor.raw_data());
                    }
                    switch (tensor.data_type())
                    {
                    case onnx::TensorProto_DataType_DOUBLE:
                        return __get_data<double>(tensor.double_data());
                    case onnx::TensorProto_DataType_FLOAT:
                    case onnx::TensorProto_DataType_FLOAT16:
                        return __get_data<double>(tensor.float_data());
                    case onnx::TensorProto_DataType_INT32:
                        return __get_data<double>(tensor.int32_data());
                    case onnx::TensorProto_DataType_INT64:
                        return __get_data<double>(tensor.int64_data());
                    case onnx::TensorProto_DataType_UINT64:
                        return __get_data<double>(tensor.uint64_data());
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid data type: "
                            << onnx::TensorProto_DataType_Name(
                                   static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
                        break;
                    }
                }

                template <>
                inline std::vector<float> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<float>(tensor.raw_data());
                    }
                    if ((tensor.data_type() == onnx::TensorProto_DataType_FLOAT) ||
                        (tensor.data_type() == onnx::TensorProto_DataType_FLOAT16))
                    {
                        return __get_data<float>(tensor.float_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return __get_data<float>(tensor.int32_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                    {
                        return __get_data<float>(tensor.int64_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                    {
                        return __get_data<float>(tensor.uint64_data());
                    }
                    NNFUSION_CHECK_FAIL()
                        << "invalid data type: "
                        << onnx::TensorProto_DataType_Name(
                               static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
                }

                template <>
                inline std::vector<int32_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<int32_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return __get_data<int32_t>(tensor.int32_data());
                    }
                    NNFUSION_CHECK_FAIL() << "invalid data type: "
                                          << onnx::TensorProto_DataType_Name(
                                                 onnx::TensorProto_DataType(tensor.data_type()));
                }

                template <>
                inline std::vector<int64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<int64_t>(tensor.raw_data());
                    }
                    NNFUSION_CHECK(tensor.data_type() == onnx::TensorProto_DataType_INT64);

                    return __get_data<int64_t>(tensor.int64_data());
                }

                template <>
                inline std::vector<uint64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return __get_raw_data<uint64_t>(tensor.raw_data());
                    }
                    NNFUSION_CHECK(tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                        << "invalid data type: "
                        << onnx::TensorProto_DataType_Name(
                               static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
                    return __get_data<uint64_t>(tensor.uint64_data());
                }

                /// \brief      Fill specified range with monotonic sequence.
                ///
                /// \param[in]  first            The iterator to the beginning of the range.
                /// \param[in]  last             The iterator to the past the end of the range.
                /// \param[in]  init_value       The initial value for sequence.
                /// \param[in]  step             The step value for sequence.
                ///
                /// \tparam     ForwardIterator  The forward iterator class type.
                /// \tparam     T                The sequence value type.
                ///
                template <typename ForwardIterator, typename T>
                inline void fill_monotonic_range(ForwardIterator first,
                                                 ForwardIterator last,
                                                 T init_value,
                                                 T step)
                {
                    for (; first != last; ++first, init_value += step)
                    {
                        *first = init_value;
                    }
                }
            }

            class Tensor;
            class Node;

            bool ONNXDataTypeToNNFusionElementType(const onnx::TensorProto_DataType onnx_dt,
                                                   nnfusion::element::Type* nnfusion_et);

            template <typename T>
            std::shared_ptr<op::Constant> make_constant_op(const element::Type& type,
                                                           const Shape shape,
                                                           const Tensor& tensor);

            std::shared_ptr<op::Constant> make_constant_op(const onnx::TensorProto_DataType onnx_et,
                                                           const Shape shape,
                                                           const Tensor& tensor);

            std::shared_ptr<graph::GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                       const onnx::NodeProto& node,
                                                       size_t input_idx);

            nnfusion::graph::GNodeIndex GetInputIndex(const NodeMap& all_ng_nodes,
                                                      const onnx::NodeProto& node,
                                                      size_t input_idx);

            graph::GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                               const onnx::NodeProto& node);

            GNodeIndexVector GetAllInputIndex(const NodeMap& all_ng_nodes,
                                              const onnx::NodeProto& node);

            /// \brief      Return the monotonic sequence.
            ///
            /// \note       Specialization for integral types.
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence.
            template <typename T,
                      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1})
            {
                std::size_t value_count = (end_value - start_value) / step;
                std::vector<T> range(value_count);
                detail::fill_monotonic_range(std::begin(range), std::end(range), start_value, step);
                return range;
            }

            /// \brief      Return the monotonic sequence.
            ///
            /// \note       Specialization for floating point types.
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence
            template <typename T,
                      typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0.f}, T step = T{1.f})
            {
                std::size_t value_count =
                    reinterpret_cast<std::size_t>(std::floor((end_value - start_value) / step));
                std::vector<T> range(value_count);
                detail::fill_monotonic_range(std::begin(range), std::end(range), start_value, step);
                return range;
            }

            /// \brief Get shape of kernel (filter) in pixels.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \param input_gnode The input gnode
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Shape get_kernel_shape(const Node& node,
                                   const std::shared_ptr<graph::GNode> input_gnode);

            /// \brief  Get number of pixels to stride operation by in each direction.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \param kernel_shape The shape of the kernel which we retrieve strides for.
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Strides get_strides(const Node& node, const Shape& kernel_shape);

            /// \brief  Get number of pixels to stride operation by in each direction.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \param input_gnode The input gnode
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Strides get_strides(const Node& node, const std::shared_ptr<graph::GNode> input_gnode);

            /// \brief Get number of pixels for filter dilation in each direction.
            ///
            /// \param node The Node ptr representing ONNX operation.
            /// \param input_gnode The input gnode
            /// \return The Strides object containing number of pixels for filter dilation
            ///         (height, width, depth).
            Strides get_dilations(const Node& node,
                                  const std::shared_ptr<graph::GNode> input_gnode);

            /// \brief Get padding values for the operation described by an ONNX node.
            /// \details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
            ///          values are calculated. Otherwise values are taken from the `pads` attribute.
            ///
            ///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
            ///
            /// \param node The Node ptr representing ONNX operation.
            /// \param kernel_shape The shape of the kernel which we retrieve pads for.
            /// \return A pair of (padding_above, padding_below), which elements contains number of
            ///         pixels to pad in respective dimensions (height, width, depth).
            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape);

            /// \brief Get padding values for the operation described by an ONNX node.
            /// \details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
            ///          values are calculated. Otherwise values are taken from the `pads` attribute.
            ///
            ///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
            ///
            /// \param node The Node ptr representing ONNX operation.
            /// \param input_gnode The input gnode
            /// \return A pair of (padding_above, padding_below), which elements contains number of
            ///         pixels to pad in respective dimensions (height, width, depth).

            inline std::pair<CoordinateDiff, CoordinateDiff>
                get_pads(const Node& node, const std::shared_ptr<graph::GNode> input_gnode)
            {
                return get_pads(node, get_kernel_shape(node, input_gnode));
            }

            CoordinateDiff get_auto_pads(const Shape& kernel_shape, const std::string& auto_pad);

            Strides get_strides_helper(const Node& node,
                                       const std::string& name,
                                       const Shape& kernel_shape);

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
