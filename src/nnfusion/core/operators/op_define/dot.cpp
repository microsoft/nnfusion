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

#include <functional>
#include <memory>
#include <utility>

#include "dot.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Dot::Dot()
    : Dot(0, false)
{
}

Dot::Dot(size_t reduction_axes_count, bool has_reduction_axes_count, bool trans_a, bool trans_b)
    : Op("Dot")
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
    , m_transpose_A(trans_a)
    , m_transpose_B(trans_b)
{
}

void Dot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    nnfusion::element::Type result_et;

    OP_VALIDATION(this,
                  nnfusion::element::Type::merge(result_et,
                                                 gnode->get_input_element_type(0),
                                                 gnode->get_input_element_type(1)))
        << "Arguments do not have the same element type (arg0 element type: "
        << gnode->get_input_element_type(0)
        << ", arg1 element type: " << gnode->get_input_element_type(1) << ").";

    const nnfusion::PartialShape& arg0_shape = gnode->get_input_partial_shape(0);
    const nnfusion::PartialShape& arg1_shape = gnode->get_input_partial_shape(1);

    // If an explicit value was not passed for reduction axis count at construction time, we have
    // some extra work to do.
    //
    // - If one of the arguments is known to be scalar, the count is 0.
    // - If both of the arguments are known to be nonscalar, the count is 1.
    // - Otherwise, the count is unknown.
    bool reduction_axes_ambiguous = !m_has_reduction_axes_count;

    if (reduction_axes_ambiguous)
    {
        if (arg0_shape.rank().same_scheme(0) || arg1_shape.rank().same_scheme(0))
        {
            m_reduction_axes_count = 0;
            reduction_axes_ambiguous = false;
        }
        else if (arg0_shape.rank().is_static() && arg1_shape.rank().is_static())
        {
            m_reduction_axes_count = 1;
            reduction_axes_ambiguous = false;
        }
    }

    nnfusion::PartialShape result_shape;

    OP_VALIDATION(this,
                  reduction_axes_ambiguous || arg0_shape.rank().is_dynamic() ||
                      m_reduction_axes_count <= size_t(arg0_shape.rank()))
        << "Reduction axes count (" << m_reduction_axes_count
        << ") is too large (arg0 shape: " << arg0_shape << ", arg1 shape: " << arg1_shape << ").";

    OP_VALIDATION(this,
                  reduction_axes_ambiguous || arg1_shape.rank().is_dynamic() ||
                      m_reduction_axes_count <= size_t(arg1_shape.rank()))
        << "Reduction axes count (" << m_reduction_axes_count
        << ") is too large (arg0 shape: " << arg0_shape << ", arg1 shape: " << arg1_shape << ").";

    if (!reduction_axes_ambiguous && arg0_shape.rank().is_static() && arg1_shape.rank().is_static())
    {
        for (size_t i = 0; i < m_reduction_axes_count; i++)
        {
            size_t axis_index_arg0 =
                m_transpose_A ? i : size_t(arg0_shape.rank()) - m_reduction_axes_count + i;
            size_t axis_index_arg1 =
                m_transpose_B ? size_t(arg1_shape.rank()) - m_reduction_axes_count + i : i;

            OP_VALIDATION(this, arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1]))
                << "Paired axes (axis " << axis_index_arg0 << " from arg0, axis " << axis_index_arg1
                << " from arg1) do not have same length (arg0 shape: " << arg0_shape
                << ", arg1 shape: " << arg1_shape
                << ", reduction axes count: " << m_reduction_axes_count
                << "transA: " << m_transpose_A << ", transB: " << m_transpose_B << ").";
        }

        std::vector<nnfusion::Dimension> result_dims(
            size_t(arg0_shape.rank()) + size_t(arg1_shape.rank()) - 2 * m_reduction_axes_count);

        size_t i = 0;

        for (size_t j = 0; j < size_t(arg0_shape.rank()) - m_reduction_axes_count; j++)
        {
            size_t idx = m_transpose_A ? size_t(arg0_shape.rank()) - 1 - j : j;
            result_dims[i++] = arg0_shape[idx];
        }
        for (size_t j = m_reduction_axes_count; j < size_t(arg1_shape.rank()); j++)
        {
            size_t idx = m_transpose_B ? size_t(arg0_shape.rank()) - 1 - j : j;
            result_dims[i++] = arg1_shape[idx];
        }

        result_shape = nnfusion::PartialShape(result_dims);
    }
    else
    {
        result_shape = nnfusion::PartialShape::dynamic();
    }

    gnode->set_output_type_and_shape(0, result_et, result_shape);
}
