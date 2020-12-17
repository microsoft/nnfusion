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

#include "pad.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Pad::Pad(const nnfusion::Shape& padding_below,
         const nnfusion::Shape& padding_above,
         const nnfusion::Shape& padding_interior)
    : Op("Pad")
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_padding_interior(padding_interior)
{
}

void Pad::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    nnfusion::element::Type result_et;

    OP_VALIDATION(this,
                  nnfusion::element::Type::merge(result_et,
                                                 gnode->get_input_element_type(0),
                                                 gnode->get_input_element_type(1)))
        << "Argument element types do not match (arg0 element type: "
        << gnode->get_input_element_type(0)
        << ", arg1 element type: " << gnode->get_input_element_type(1) << ").";

    OP_VALIDATION(this, gnode->get_input_partial_shape(1).compatible(nnfusion::PartialShape{}))
        << "Argument for padding value is not a scalar (shape: "
        << gnode->get_input_partial_shape(1) << ").";

    auto arg_shape = gnode->get_input_partial_shape(0);

    OP_VALIDATION(this,
                  m_padding_below.size() == m_padding_above.size() &&
                      m_padding_below.size() == m_padding_interior.size())
        << "Ranks for padding below (" << m_padding_below << "), padding above (" << m_padding_above
        << ") and interior padding (" << m_padding_interior << ") "
        << "do not match.";

    size_t implied_rank = m_padding_below.size();

    OP_VALIDATION(this, arg_shape.rank().compatible(implied_rank))
        << "Rank for padding below/padding above/interior padding does not match the rank of the "
        << "data argument (padding below: " << m_padding_below << ", "
        << ", padding above: " << m_padding_above << ", interior padding: " << m_padding_interior
        << ").";

    std::vector<nnfusion::Dimension> result_dims(implied_rank, nnfusion::Dimension::dynamic());

    if (arg_shape.rank().is_static())
    {
        for (size_t i = 0; i < implied_rank; i++)
        {
            if (arg_shape[i].is_static())
            {
                result_dims[i] =
                    m_padding_below[i] +
                    subtract_or_zero(size_t(arg_shape[i]) * (m_padding_interior[i] + 1),
                                     m_padding_interior[i]) +
                    m_padding_above[i];
            }
        }
    }

    gnode->set_output_type_and_shape(0, result_et, PartialShape(result_dims));
}
