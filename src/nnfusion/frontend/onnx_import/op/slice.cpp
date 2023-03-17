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

#include <unordered_set>
#include <vector>

#include "../util/util.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"
#include "slice.hpp"

DEFINE_bool(ffix_slice, false, "A hack fix to re-pass symbolic values to slice op");

static inline int64_t get_valid_array_idx(int64_t idx, int64_t last_idx)
{
    return (idx >= 0) ? std::min(idx, last_idx) : std::max<int64_t>(0, last_idx + idx);
}

static inline void
    processSliceInputs(const int64_t input_rank, int64_t& start, int64_t& end, int64_t& step)
{
    auto clamp = [](int64_t val, int64_t min, int64_t max) -> int64_t {
        return (val < min) ? min : (val > max) ? max : val;
    };
    // process step
    NNFUSION_CHECK(step != 0);
    // process start
    if (start < 0)
        start += input_rank;
    if (step < 0)
        start = clamp(start, 0, input_rank - 1);
    else
        start = clamp(start, 0, input_rank);
    // process end
    if (end < 0)
        end += input_rank;
    if (step < 0)
        end = clamp(end, -1, input_rank - 1);
    else
        end = clamp(end, 0, input_rank);
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
                    for (auto i = 0; i < axes.size(); i++)
                    {
                        axes[i] += axes[i] < 0 ? data.get_shape().size() : 0;
                    }

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
                    nnfusion::json stat;
                    stat["starts"] = starts;
                    stat["ends"] = ends;
                    stat["axes"] = axes;
                    std::vector<int64_t> steps;
                    steps.resize(starts.size(), 1);
                    stat["steps"] = steps;
                    op->deserialize(stat);
                    // NNFUSION_LOG(INFO) << stat.dump();
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
                    NNFUSION_CHECK(starts.size() == ends.size());
                    std::vector<int64_t> axes;
                    if (inputs.size() > 3)
                    {
                        NNFUSION_CHECK(GetValueFromNGraphOp(inputs[3].gnode, &axes));
                        for (auto i = 0; i < axes.size(); i++)
                        {
                            axes[i] += axes[i] < 0 ? data.get_shape().size() : 0;
                        }
                    }
                    else
                    {
                        axes.resize(starts.size());
                        std::iota(axes.begin(), axes.end(), 0);
                    }
                    NNFUSION_CHECK(axes.size() == starts.size());

                    std::vector<int64_t> steps;
                    if (inputs.size() > 4)
                    {
                        NNFUSION_CHECK(GetValueFromNGraphOp(inputs[4].gnode, &steps));
                    }
                    else
                    {
                        steps.resize(starts.size(), 1);
                    }
                    NNFUSION_CHECK(steps.size() == axes.size());

                    Shape data_shape = data.get_shape();
                    size_t data_rank = data_shape.size();
                    Shape lower_bounds(data_rank, 0);
                    Shape upper_bounds = data_shape;
                    Strides strides(data_rank, 1);
                    std::vector<int64_t> steps_op(data_rank, 1);
                    Shape out_shape = data_shape;

                    std::unordered_set<int64_t> unique_axes;
                    for (size_t idx = 0; idx < axes.size(); ++idx)
                    {
                        int64_t axis = axes.at(idx) < 0 ? axes.at(idx) + data_rank : axes.at(idx);
                        NNFUSION_CHECK(axis >= 0 && axis < data_rank);
                        NNFUSION_CHECK(unique_axes.find(axis) == unique_axes.end());
                        unique_axes.insert(axis);

                        int64_t start = starts.at(idx);
                        int64_t end = ends.at(idx);
                        int64_t step = steps.at(idx);
                        int64_t data_dim = data_shape.at(axis);
                        processSliceInputs(data_dim, start, end, step);

                        lower_bounds.at(axis) = start;
                        upper_bounds.at(axis) = end;
                        strides.at(axis) = abs(step);
                        steps_op.at(axis) = step;
                    }

                    for (auto i = 0; i < data_shape.size(); i++)
                    {
                        size_t result_axis_size = upper_bounds[i] - lower_bounds[i];
                        result_axis_size = result_axis_size / strides[i] +
                                           ((result_axis_size % strides[i] == 0) ? 0 : 1);
                        out_shape[i] = result_axis_size;
                    }

                    if (FLAGS_ffix_slice)
                    {
                        if (data_shape == Shape{2048, 2048} && axes.size() == 1 && axes[0] == 0)
                        {
                            auto dim_params = m_graph->get_dim_params();
                            if (dim_params.count("past_seq") > 0 &&
                                dim_params["past_seq"].is_dynamic())
                            {
                                NNFUSION_CHECK(dim_params.count("seq") > 0 &&
                                               dim_params["deq"].is_static());
                                SymDim past_seq = dim_params["past_seq"];
                                SymDim seq = dim_params["seq"];
                                auto symbol_lower_bounds = std::make_shared<SymShape>(lower_bounds);
                                auto symbol_upper_bounds = std::make_shared<SymShape>(upper_bounds);
                                auto symbol_out_shape = std::make_shared<SymShape>(out_shape);
                                NNFUSION_CHECK(lower_bounds[0] == past_seq.max());
                                NNFUSION_CHECK(upper_bounds[0] == (past_seq.max() + seq.max()));
                                symbol_lower_bounds->at(0) = past_seq;
                                symbol_upper_bounds->at(0) = (past_seq + seq);
                                lower_bounds.set_sym_shape(symbol_lower_bounds);
                                upper_bounds.set_sym_shape(symbol_upper_bounds);
                                out_shape.set_sym_shape(symbol_out_shape);
                                NNFUSION_LOG(INFO) << "Fix Slice symbol bounds: " << lower_bounds
                                                   << " | " << upper_bounds;
                            }
                        }

                        if (data_shape.size() == 2 && data_shape[1] == 2048 && axes.size() == 1 &&
                            axes[0] == 1)
                        {
                            auto dim_params = m_graph->get_dim_params();
                            if (dim_params.count("past_seq") > 0 &&
                                dim_params["past_seq"].is_dynamic())
                            {
                                NNFUSION_CHECK(dim_params.count("seq") > 0 &&
                                               dim_params["deq"].is_static());
                                SymDim past_seq = dim_params["past_seq"];
                                SymDim seq = dim_params["seq"];
                                auto symbol_lower_bounds = std::make_shared<SymShape>(lower_bounds);
                                auto symbol_upper_bounds = std::make_shared<SymShape>(upper_bounds);
                                auto symbol_out_shape = std::make_shared<SymShape>(out_shape);
                                NNFUSION_CHECK(lower_bounds[1] == 0);
                                NNFUSION_CHECK(upper_bounds[1] == (past_seq.max() + seq.max()));
                                symbol_upper_bounds->at(1) = (past_seq + seq);
                                symbol_out_shape->at(1) = (past_seq + seq);
                                lower_bounds.set_sym_shape(symbol_lower_bounds);
                                upper_bounds.set_sym_shape(symbol_upper_bounds);
                                out_shape.set_sym_shape(symbol_out_shape);
                                NNFUSION_LOG(INFO) << "Fix Slice symbol bounds: " << lower_bounds
                                                   << " | " << upper_bounds;
                            }
                        }
                    }

                    auto op = std::make_shared<op::Slice>(
                        lower_bounds, upper_bounds, strides, out_shape, steps_op);
                    op->set_name(node_proto.output(0));
                    nnfusion::json stat;
                    stat["starts"] = starts;
                    stat["ends"] = ends;
                    stat["axes"] = axes;
                    stat["steps"] = steps;
                    op->deserialize(stat);
                    auto gnode = m_graph->add_node_and_edge(op, {data});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_10

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
