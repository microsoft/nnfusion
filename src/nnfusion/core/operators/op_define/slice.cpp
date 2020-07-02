// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "slice.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Slice::Slice(const nnfusion::Coordinate& lower_bounds,
             const nnfusion::Coordinate& upper_bounds,
             const nnfusion::Strides& strides)
    : Op("Slice")
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
}

Slice::Slice(const nnfusion::Coordinate& lower_bounds, const nnfusion::Coordinate& upper_bounds)
    : Op("Slice")
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(nnfusion::Strides())
{
}

void Slice::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // An empty stride vector with lower_bounds/upper_bounds filled in means that we need to
    // construct the default value.
    if (m_strides.size() == 0)
    {
        m_strides = nnfusion::Strides(m_lower_bounds.size(), 1);
    }

    OP_VALIDATION(this,
                  m_lower_bounds.size() == m_upper_bounds.size() &&
                      m_lower_bounds.size() == m_strides.size())
        << "Ranks of lower bounds (" << m_lower_bounds << "), upper bounds (" << m_upper_bounds
        << ") and strides (" << m_strides << ") do not match.";

    size_t output_rank = m_upper_bounds.size();

    for (size_t i = 0; i < output_rank; i++)
    {
        OP_VALIDATION(this, m_lower_bounds[i] <= m_upper_bounds[i])
            << "Lower bound for slice is greater than upper bound at axis " << i
            << " (lower bounds: " << m_lower_bounds << ", upper bounds: " << m_upper_bounds << ").";

        OP_VALIDATION(this, m_strides[i] != 0) << "Stride for slice is zero at axis " << i
                                               << " (strides: " << m_strides << ").";
    }

    const nnfusion::PartialShape& input_shape = gnode->get_input_partial_shape(0);
    nnfusion::Dimension input_rank = input_shape.rank();

    OP_VALIDATION(this, input_rank.is_dynamic() || size_t(input_rank) == output_rank)
        << "Input rank does not match the rank of the lower bounds (" << m_lower_bounds
        << "), upper bounds (" << m_upper_bounds << "), and strides (" << m_strides << ").";

    std::vector<nnfusion::Dimension> result_dims(output_rank);

    for (size_t i = 0; i < output_rank; i++)
    {
        OP_VALIDATION(this,
                      input_rank.is_dynamic() || input_shape[i].is_dynamic() ||
                          m_upper_bounds[i] <= size_t(input_shape[i]))
            << "Upper bound for slice at axis " << i << " is out of range "
            << "(upper bounds: " << m_upper_bounds << ", argument shape: " << input_shape << ").";

        size_t result_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        result_axis_size =
            result_axis_size / m_strides[i] + ((result_axis_size % m_strides[i] == 0) ? 0 : 1);
        result_dims[i] = result_axis_size;
    }

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), nnfusion::PartialShape{result_dims});
}
