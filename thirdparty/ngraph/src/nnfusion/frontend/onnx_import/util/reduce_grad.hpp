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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "nnfusion/common/common.hpp"
#include "op/shape.hpp"
#include "reshape.hpp"
#include "util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace reduce
            {
                using GNode = nnfusion::graph::GNode;

                inline AxisSet get_reduction_axes(const Shape& in_shape, const Shape& out_shape)
                {
                    AxisSet reduction_axes{};
                    NNFUSION_CHECK(out_shape.size() <= in_shape.size());
                    auto rank_diff = in_shape.size() - out_shape.size();

                    for (auto axis = 0; axis < in_shape.size(); axis++)
                    {
                        if (axis <= rank_diff)
                        {
                            reduction_axes.insert(axis);
                        }
                        else
                        {
                            NNFUSION_CHECK(out_shape[axis - rank_diff] == 1 ||
                                           out_shape[axis - rank_diff] == in_shape[axis]);
                            if (out_shape[axis - rank_diff] != in_shape[axis])
                            {
                                reduction_axes.insert(axis);
                            }
                        }
                    }
                    return reduction_axes;
                }

                template <class OpType = op::Sum,
                          typename = typename std::enable_if<
                              std::is_base_of<nnfusion::op::ArithmeticReduction, OpType>::value>>
                GNodeIndex reduce_grad(const GNodeIndex input,
                                       const Shape& out_shape,
                                       std::shared_ptr<nnfusion::graph::Graph> graph)
                {
                    auto input_shape = input.get_shape();
                    if (input_shape == out_shape)
                    {
                        return input;
                    }

                    auto reduction_axes = get_reduction_axes(input_shape, out_shape);

                    auto reduce_op = std::make_shared<OpType>(reduction_axes);
                    auto reduce_gnode = graph->add_node_and_edge(reduce_op, {input});

                    auto reshape_op = std::make_shared<op::Reshape>(
                        nnfusion::frontend::onnx_import::reshape::get_default_axis_vector(
                            reduce_gnode->get_shape().size()),
                        out_shape);
                    auto reshape_gnode = graph->add_node_and_edge(reshape_op, {reduce_gnode});

                    return GNodeIndex{reshape_gnode};
                }

            } // namespace  reduce
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
