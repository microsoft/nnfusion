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

#include "recursion.hpp"

using namespace std;
using namespace nnfusion::op;

Recursion::Recursion(std::shared_ptr<nnfusion::graph::Graph>& body_graph,
                     const std::vector<nnfusion::PartialShape>& output_shapes,
                     const std::vector<nnfusion::element::Type>& output_types)
    : Op("Recursion")
    , m_body_graph(body_graph)
    , m_output_shapes(output_shapes)
    , m_output_types(output_types)
{
}

FuncForward::FuncForward()
    : Op("FuncForward")
{
}

void Recursion::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    for (size_t i = 0; i < gnode->get_output_size(); i++)
    {
        gnode->set_output_type_and_shape(i, m_output_types[i], m_output_shapes[i]);
    }
}
