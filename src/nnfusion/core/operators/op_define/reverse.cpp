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

#include <algorithm>
#include <sstream>

#include "nnfusion/core/graph/gnode.hpp"
#include "reverse.hpp"

using namespace std;
using namespace nnfusion::op;

Reverse::Reverse(const nnfusion::AxisSet& reversed_axes)
    : Op("Reverse")
    , m_reversed_axes(reversed_axes)
{
}

void Reverse::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_shape = gnode->get_input_partial_shape(0);
    nnfusion::Dimension input_rank = input_shape.rank();

    if (input_rank.is_static())
    {
        // Make sure all reversed axis indices are valid.
        for (size_t axis : m_reversed_axes)
        {
            OP_VALIDATION(this, axis < size_t(input_rank))
                << "Reverse axis (" << axis << ") is out of bounds (argument shape: " << input_shape
                << ").";
        }
    }

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape);
}