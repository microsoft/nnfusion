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

#include <memory>

#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"

#include "../util/util.hpp"
#include "nnfusion/common/axis_vector.hpp"
#include "nnfusion/common/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            /// \brief Cast shape of two nodes to make them compatible for an element-wise binary operation.
            ///
            /// If necessary the right-hand-side argument will be broadcast to match the shape
            /// of left-hand-side argument. The starting of the mutually equal shape is
            /// specified by the argument "start_match_axis", and if it is not set,
            /// suffix matching is assumed.
            ///
            /// This style of broadcast was used in ONNX Op sets prior to version 7, where it was
            /// replaced by numpy-style broadcasting.
            ///
            /// \param left Node which contain input of binary op.
            /// \param right Node which contain input of binary op.
            /// \param start_match_axis position in shape denoting start of the mutually equal shape
            ///
            /// \return Left and right node after broadcasting.
            std::pair<std::shared_ptr<graph::GNode>, std::shared_ptr<graph::GNode>>
                legacy_style_broadcast_for_binary_operation(
                    const std::pair<std::shared_ptr<graph::GNode>, std::shared_ptr<graph::GNode>>&
                        args,
                    std::size_t start_match_axis,
                    std::shared_ptr<nnfusion::graph::Graph> graph);

            /// \brief      Broadcast shape of two nodes to make them compatible for a matrix multiplication.
            ///
            /// \note       This function is reflecting broadcasting behaviour of NumPys' `matmul` operation
            ///             \link https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
            ///             This mean that only \"stack of matrices\" axes are bidirectionally broadcasted.
            ///             The last two dimension are left untouched.
            ///
            /// \param[in]  left   The Node providing data for the left-hand side of matrix multiplication.
            /// \param[in]  right  The Node providing data for the right-hand side of matrix multiplication.
            ///
            /// \return     The vector containing both nodes broadcasted.
            ///
            graph::GNodeVector numpy_style_broadcast_for_matmul_operation(
                const std::shared_ptr<graph::GNode>& left,
                const std::shared_ptr<graph::GNode>& right,
                std::shared_ptr<nnfusion::graph::Graph> graph);

            /// \brief Generate a list of broadcast axes.
            ///
            /// \details Informally, a broadcast "adds" axes to the input tensor, replicating
            ///          elements from the input tensor as needed to fill the new dimensions.
            ///          Function calculate which of the output axes are added in this way.
            ///
            /// \param output_shape      The new shape for the output tensor.
            /// \param input_shape       The shape of input tensor.
            /// \param start_match_axis  The axis along which we want to replicate elements.
            ///                          The starting axis position (0-based) int the output
            ///                          shape from which the current shape of the tensor
            ///                          matches the desired new shape.
            ///
            /// \return The indices of added axes.
            AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                             const Shape& input_shape,
                                             std::size_t start_match_axis);

            /// \brief Generate a list of broadcast along axes.
            ///
            /// \details Broadcast "adds" elements along axes to the input tensor, replicating
            ///          elements from the input tensor as needed to fill the new dimensions.
            ///          Function calculate which of the output axes are added in this way.
            ///
            ///          This function will attempt to match shapes, assuming the current shape
            ///          matches the rightmost positions of the desired new shape. This behaviour
            ///          is similar to NumPy's broadcasting.
            ///
            /// \param output_shape The new shape for the output tensor.
            /// \param input_shape  The shape of input tensor.
            ///
            /// \return             The indices of added axes.
            inline AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                                    const Shape& input_shape)
            {
                return calculate_broadcast_axes(
                    output_shape, input_shape, output_shape.size() - input_shape.size());
            }

            inline std::shared_ptr<graph::GNode>
                make_broadcast_node(const std::shared_ptr<graph::GNode>& gnode,
                                    Shape new_shape,
                                    std::shared_ptr<nnfusion::graph::Graph> graph)
            {
                auto return_op = std::make_shared<op::Broadcast>(
                    new_shape, calculate_broadcast_axes(new_shape, gnode->get_shape()));
                return graph->add_node_and_edge(return_op, {gnode});
            }

            inline std::shared_ptr<graph::GNode>
                make_broadcast_node(const std::shared_ptr<graph::GNode>& gnode,
                                    Shape new_shape,
                                    std::size_t start_match_axis,
                                    std::shared_ptr<nnfusion::graph::Graph> graph)
            {
                auto return_op = std::make_shared<op::Broadcast>(
                    new_shape,
                    calculate_broadcast_axes(new_shape, gnode->get_shape(), start_match_axis));
                return graph->add_node_and_edge(return_op, {gnode});
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
