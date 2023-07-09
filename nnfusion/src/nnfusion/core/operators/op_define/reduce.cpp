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

#include "reduce.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::op;

Reduce::Reduce(const shared_ptr<graph::Graph>& reduction_graph,
               const nnfusion::AxisSet& reduction_axes)
    : Op("Reduce")
    , m_reduction_graph(reduction_graph)
    , m_reduction_axes(reduction_axes)
{
}

void Reduce::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_reductee = gnode->get_inputs().at(0);

    auto input_init = gnode->get_inputs().at(1);
    NNFUSION_CHECK(input_init->get_shape().size() == 0)
        << "Argument for initial value is not a scalar";

    NNFUSION_CHECK(input_init->get_element_type() == input_reductee->get_element_type())
        << "Element types for reductee and initial values do not match";

    auto input_reductee_shape = input_reductee->get_shape();

    for (auto axis : m_reduction_axes)
    {
        NNFUSION_CHECK(axis < input_reductee_shape.size()) << "Reduction axis is out of bounds";
    }

    nnfusion::Shape result_shape;

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(input_reductee_shape.at(i));
        }
    }

    auto g_params = m_reduction_graph->get_parameters();
    auto arg_reductee = gnode->get_in_edge(0)->get_src();
    auto arg_init = gnode->get_in_edge(1)->get_src();
    NNFUSION_CHECK(g_params.size() == 2)
        << "Reduction graph has wrong number of parameters (should be two)";

    NNFUSION_CHECK(g_params.at(0)->has_same_type(arg_init))
        << "Argument 0 of reduction graph has wrong type";
    NNFUSION_CHECK(g_params.at(1)->has_same_type(arg_init))
        << "Argument 1 of reduction graph has wrong type";

    NNFUSION_CHECK(m_reduction_graph->get_output_size() == 1)
        << "Single-output reduce graph was expected!";

    NNFUSION_CHECK(m_reduction_graph->get_outputs().at(0)->get_element_type() ==
                   arg_init->get_element_type())
        << "Return element type from reduction graph does not match expected";
    NNFUSION_CHECK(m_reduction_graph->get_outputs().at(0)->get_shape() == nnfusion::Shape{})
        << "Return shape from reduction graph is not a scalar";
    gnode->set_output_type_and_shape(0, input_reductee->get_element_type(), result_shape);
}
