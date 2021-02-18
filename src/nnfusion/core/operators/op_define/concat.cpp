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

#include <cassert>
#include <memory>

#include "concat.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Concat::Concat(size_t concatenation_axis)
    : Op("Concat")
    , m_concatenation_axis(concatenation_axis)
{
}

void Concat::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this, gnode->get_input_size() >= 1) << "At least one argument required.";

    nnfusion::PartialShape inputs_shape_scheme{nnfusion::PartialShape::dynamic()};
    nnfusion::element::Type inputs_et{nnfusion::element::dynamic};
    nnfusion::Dimension concatenation_axis_output_dim{0};

    for (auto i = 0; i < gnode->get_input_size(); i++)
    {
        nnfusion::PartialShape this_input_shape = gnode->get_input_partial_shape(i);
        nnfusion::Dimension this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static())
        {
            OP_VALIDATION(this, m_concatenation_axis < size_t(this_input_rank))
                << "Concatenation axis (" << m_concatenation_axis << ") is out of bounds for "
                << "argument " << i << ", which has shape " << this_input_shape << ".";

            concatenation_axis_output_dim += this_input_shape[m_concatenation_axis];
            this_input_shape[m_concatenation_axis] = nnfusion::Dimension::dynamic();

            OP_VALIDATION(this,
                          nnfusion::PartialShape::merge_into(inputs_shape_scheme, this_input_shape))
                << "Argument shapes are inconsistent; they must have the same rank, and must have "
                << "equal dimension everywhere except on the concatenation axis (axis "
                << m_concatenation_axis << ").";

            OP_VALIDATION(this,
                          nnfusion::element::Type::merge(
                              inputs_et, inputs_et, gnode->get_input_element_type(i)))
                << "Argument element types are inconsistent.";
        }
        else
        {
            concatenation_axis_output_dim += nnfusion::Dimension::dynamic();
        }
    }

    nnfusion::PartialShape concatenated_shape = inputs_shape_scheme;

    if (concatenated_shape.rank().is_static())
    {
        concatenated_shape[m_concatenation_axis] = concatenation_axis_output_dim;
    }

    gnode->set_output_type_and_shape(0, inputs_et, concatenated_shape);
}

void Concat::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_shape(0);
    auto& output_shape = gnode->get_output_shape(0);
    if (input_shape.size() == output_shape.size())
    {
        m_shared_memory.clear();
        for (size_t i = 0; i < output_shape.size(); i++)
        {
            if (i != get_concatenation_axis())
                m_shared_memory.push_back(1);
            else
                m_shared_memory.push_back(output_shape[i]);
        }
    }
}
