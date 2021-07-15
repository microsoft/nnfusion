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

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"

using namespace std;
using namespace nnfusion::op;

Loop::Loop(std::shared_ptr<nnfusion::graph::Graph>& loop_body_graph,
           const std::vector<nnfusion::PartialShape>& output_shapes,
           const std::vector<nnfusion::element::Type>& output_types)
    : Op("Loop")
    , m_loop_body_graph(loop_body_graph)
    , m_output_shapes(output_shapes)
    , m_output_types(output_types)
{
}

void Loop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    nnfusion::Shape trip_shape = gnode->get_input_shape(0);
    nnfusion::element::Type trip_et = gnode->get_input_element_type(0);
    NNFUSION_CHECK(trip_shape.size() == 0)
        << "The trip-count tensor of the Loop operation mush be scalar.";
    NNFUSION_CHECK(trip_et == nnfusion::element::i64)
        << "The trip-count tensor of the Loop operation mush be boolean.";

    nnfusion::Shape cond_shape = gnode->get_input_shape(1);
    nnfusion::element::Type cond_et = gnode->get_input_element_type(1);
    NNFUSION_CHECK(cond_shape.size() == 0)
        << "The condition tensor of the Loop operation mush be scalar.";
    NNFUSION_CHECK(cond_et == nnfusion::element::boolean)
        << "The condition tensor of the Loop operation mush be boolean.";

    for (size_t i = 0; i < gnode->get_output_size(); i++)
    {
        gnode->set_output_type_and_shape(i, m_output_types[i], m_output_shapes[i]);
    }
}