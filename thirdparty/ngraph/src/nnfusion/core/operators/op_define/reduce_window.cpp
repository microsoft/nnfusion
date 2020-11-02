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

// Microsoft (c) 2019, NNFusion Team

#include "reduce_window.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::op;

ReduceWindow::ReduceWindow(const std::shared_ptr<graph::Graph>& reduction_graph,
                           const nnfusion::Shape& window_shape,
                           const nnfusion::Strides& window_movement_strides)
    : Op("ReduceWindow")
    , m_reduction_graph(reduction_graph)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
{
}

void ReduceWindow::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_reductee = gnode->get_inputs().at(0);
    auto input_init = gnode->get_inputs().at(1);
    auto input_reductee_shape = input_reductee->get_shape();
    auto input_init_shape = input_init->get_shape();

    NNFUSION_CHECK(input_init->get_shape().size() == 0)
        << "Argument for initial value is not a scalar";

    NNFUSION_CHECK(input_init->get_element_type() == input_reductee->get_element_type())
        << "Element types for reductee and initial values do not match";

    NNFUSION_CHECK(input_reductee_shape.size() == m_window_shape.size())
        << "Window shape has different rank from input tensor";

    NNFUSION_CHECK(input_reductee_shape.size() == m_window_movement_strides.size())
        << "Window movement strides have different rank from input tensor";

    for (size_t s : m_window_shape)
    {
        NNFUSION_CHECK(s != 0) << "Window shape has a zero-length axis";
    }

    for (size_t s : m_window_movement_strides)
    {
        NNFUSION_CHECK(s != 0) << "Window movement stride for some axis is zero";
    }

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        NNFUSION_CHECK(m_window_shape[i] <= input_reductee_shape[i])
            << "Reduction window is bigger than input";
    }

    auto g_params = m_reduction_graph->get_parameters();
    auto arg_reductee = gnode->get_in_edge(0)->get_src();
    auto arg_init = gnode->get_in_edge(1)->get_src();
    NNFUSION_CHECK(g_params.size() == 2)
        << "Reduction graph has wrong number of parameters (should be two)";

    NNFUSION_CHECK(g_params.at(0)->get_element_type() == arg_init->get_element_type())
        << "Parameter 0 of reduction graph has wrong element type";

    NNFUSION_CHECK(g_params.at(1)->get_element_type() == arg_init->get_element_type())
        << "Parameter 1 of reduction graph has wrong element type";

    NNFUSION_CHECK(g_params.at(0)->get_shape() == nnfusion::Shape{})
        << "Parameter 0 of reduction graph is not a scalar";

    NNFUSION_CHECK(g_params.at(1)->get_shape() == nnfusion::Shape{})
        << "Parameter 1 of reduction graph is not a scalar";

    NNFUSION_CHECK(m_reduction_graph->get_output_size() <= 1)
        << "Single-output reduction graph was expected";

    NNFUSION_CHECK(m_reduction_graph->get_outputs().at(0)->get_element_type() ==
                   arg_init->get_element_type())
        << "Return element type from reduction graph does not match expected";

    NNFUSION_CHECK(m_reduction_graph->get_outputs().at(0)->get_shape() == nnfusion::Shape{})
        << "Return shape from reduction graph is not a scalar";

    nnfusion::Shape result_shape;

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        result_shape.push_back(ceil_div(input_reductee_shape[i] - m_window_shape[i] + 1,
                                        m_window_movement_strides[i]));
    }

    gnode->set_output_type_and_shape(0, input_reductee->get_element_type(), result_shape);
}