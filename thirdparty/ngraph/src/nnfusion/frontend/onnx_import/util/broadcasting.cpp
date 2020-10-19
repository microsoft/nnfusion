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

#include <iterator>
#include <numeric>
#include <vector>

#include "broadcasting.hpp"
#include "reshape.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            /// \brief Calculate output shape of numpy - style broadcast operation.
            ///        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
            ///
            /// \param left_shape Shape of first input tensor.
            /// \param right_shape Shape of the second input tensor.
            /// \return Shape of the output tensor and full shape of input tensors.
            std::vector<Shape> get_numpy_broadcast_shape(Shape left_shape, Shape right_shape)
            {
                Shape output_shape;
                auto rank_left = left_shape.size();
                auto rank_right = right_shape.size();
                auto max_rank = std::max(rank_left, rank_right);

                for (auto i = 0; i < (max_rank - rank_left); ++i)
                {
                    left_shape.insert(std::begin(left_shape), 1);
                }
                for (auto i = 0; i < (max_rank - rank_right); ++i)
                {
                    right_shape.insert(std::begin(right_shape), 1);
                }
                for (auto index = 0; index < max_rank; ++index)
                {
                    output_shape.push_back(std::max(left_shape.at(index), right_shape.at(index)));
                }

                return {output_shape, left_shape, right_shape};
            }

            /// \brief      Broadcast input node.
            ///
            /// \note       The source shape does not have to be the actual shape of input node. However
            ///             it should be a superset of it (containing it as a continuous subset). This implies
            ///             we may expand the number of axes of input node.
            ///
            /// \param[in]  node          The input Node to be broadcasted.
            /// \param[in]  output_shape  The output shape.
            /// \param[in]  source_shape  The source shape from which we want to broadcast input node.
            ///
            /// \return     The boroadcasted Node.
            ///
            static std::shared_ptr<graph::GNode>
                broadcast(const std::shared_ptr<graph::GNode>& node,
                          const Shape& output_shape,
                          const Shape& source_shape,
                          std::shared_ptr<nnfusion::graph::Graph> graph)
            {
                AxisVector broadcast_axes;
                Shape squeezed_shape;
                // Positions of axes which have length of 1 are needed to calculate broadcast_axes
                // for nGraph broadcast operation. We need to remove all ones from source shape
                // to avoid broadcasting axis conflict.
                for (std::size_t index = 0; index < output_shape.size(); ++index)
                {
                    if (source_shape.at(index) == 1)
                    {
                        broadcast_axes.push_back(index);
                    }
                    else
                    {
                        squeezed_shape.push_back(source_shape.at(index));
                    }
                }

                // Remove axes which have length of 1 from source shape
                auto reshape_op = std::make_shared<op::Reshape>(
                    reshape::get_default_axis_vector(node->get_shape().size()), squeezed_shape);

                auto reshape_gnode = graph->add_node_and_edge(reshape_op, {node});
                auto broadcasted_op = std::make_shared<op::Broadcast>(output_shape, broadcast_axes);
                auto broadcasted_gnode = graph->add_node_and_edge(broadcasted_op, {reshape_gnode});
                return broadcasted_gnode;
            }
            /*
            NodeVector
                numpy_style_broadcast_for_matmul_operation(const std::shared_ptr<graph::GNode>& left,
                                                        const std::shared_ptr<graph::GNode>& right)
            {
                const auto& left_shape = left->get_shape();
                const auto& right_shape = right->get_shape();
                // Broadcast only _stack of matrices_ axes.
                const auto& numpy_shapes = get_numpy_broadcast_shape(
                    Shape{std::begin(left_shape), std::next(std::end(left_shape), -2)},
                    Shape{std::begin(right_shape), std::next(std::end(right_shape), -2)});

                // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
                auto left_output_shape = numpy_shapes.at(0);
                auto right_output_shape = numpy_shapes.at(0);
                // Append the last two axes original dimensions.
                left_output_shape.insert(std::end(left_output_shape),
                                        std::next(std::begin(left_shape), left_shape.size() - 2),
                                        std::end(left_shape));
                right_output_shape.insert(std::end(right_output_shape),
                                        std::next(std::begin(right_shape), right_shape.size() - 2),
                                        std::end(right_shape));

                auto left_full_shape = numpy_shapes.at(1);
                auto right_full_shape = numpy_shapes.at(2);
                // Append the last two axes original dimensions.
                left_full_shape.insert(std::end(left_full_shape),
                                    std::next(std::begin(left_shape), left_shape.size() - 2),
                                    std::end(left_shape));
                right_full_shape.insert(std::end(right_full_shape),
                                        std::next(std::begin(right_shape), right_shape.size() - 2),
                                        std::end(right_shape));

                return {broadcast(left, left_output_shape, left_full_shape),
                        broadcast(right, right_output_shape, right_full_shape)};
            }
            */
            std::pair<std::shared_ptr<graph::GNode>, std::shared_ptr<graph::GNode>>
                legacy_style_broadcast_for_binary_operation(
                    const std::pair<std::shared_ptr<graph::GNode>, std::shared_ptr<graph::GNode>>&
                        args,
                    std::size_t start_match_axis,
                    std::shared_ptr<nnfusion::graph::Graph> graph)
            {
                const auto& left_shape = args.first->get_shape();
                const auto& right_shape = args.second->get_shape();

                if (left_shape == right_shape)
                {
                    return args;
                }

                // Prepare new shape of right operand for broadcasting
                // Remove dimensions with length=1 from back
                auto new_right_shape = right_shape;
                for (int dimension = new_right_shape.size() - 1; dimension >= 0; --dimension)
                {
                    if (new_right_shape[dimension] == 1)
                    {
                        new_right_shape.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }

                // Find first dimensions at front with length different from 1
                size_t num_ones = 0;
                for (size_t dimension : new_right_shape)
                {
                    if (dimension == 1)
                    {
                        ++num_ones;
                    }
                    else
                    {
                        break;
                    }
                }

                // Remove dimensions with length=1 from front
                new_right_shape.erase(std::begin(new_right_shape),
                                      std::next(std::begin(new_right_shape), num_ones));

                auto reshape_right_op = std::make_shared<op::Reshape>(
                    reshape::get_default_axis_vector(right_shape.size()), new_right_shape);
                auto reshape_right_gnode =
                    graph->add_node_and_edge(reshape_right_op, {args.second});

                // Move broadcast start axis parameter to right
                start_match_axis += num_ones;

                auto broadcast_right_op = std::make_shared<op::Broadcast>(
                    left_shape,
                    calculate_broadcast_axes(left_shape, new_right_shape, start_match_axis));
                auto broadcast_right_gnode =
                    graph->add_node_and_edge(broadcast_right_op, {reshape_right_gnode});
                return {args.first, broadcast_right_gnode};
            }

            AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                             const Shape& input_shape,
                                             std::size_t start_match_axis)
            {
                std::vector<size_t> result(output_shape.size() - input_shape.size());
                // Populate the result vector with monotonic increasing series from 0 until
                // output_shape_size, excluding values in range [start_match_axis, start_match_axis + input_shape.size()
                std::iota(std::begin(result), std::begin(result) + start_match_axis, 0);
                std::iota(std::begin(result) + start_match_axis,
                          std::end(result),
                          start_match_axis + input_shape.size());
                return result;
            }

        } // namespace  onnx_import
    }     // namespace frontend
} // namespace  nnfusion
