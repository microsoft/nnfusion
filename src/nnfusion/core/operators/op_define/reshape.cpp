// Microsoft (c) 2019, NNFusion Team

#include <algorithm>
#include <iostream>

#include "nnfusion/core/graph/gnode.hpp"
#include "reshape.hpp"

using namespace std;
using namespace nnfusion::op;

Reshape::Reshape(const nnfusion::AxisVector& input_order, const nnfusion::Shape& output_shape)
    : Op("Reshape")
    , m_input_order(input_order)
    , m_output_shape(output_shape)
{
}

void Reshape::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    // Check that the input axis order is a permutation of (0,...,n-1) for some n.
    for (size_t i = 0; i < m_input_order.size(); i++)
    {
        OP_VALIDATION(this, find(begin(m_input_order), end(m_input_order), i) != end(m_input_order))
            << "Input axis order is not a permutation of argument's axis indices (axis order: "
            << m_input_order << ", argument shape: " << input_shape << ").";
    }

    // TODO(amprocte): should be possible to move around unknown dims in the input shape.
    if (input_rank.is_static())
    {
        OP_VALIDATION(this, m_input_order.size() == size_t(input_rank))
            << "Input axis order is not a permutation of argument's axis indices (axis order: "
            << m_input_order << ", argument shape: " << input_shape << ").";

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            auto it = find(begin(m_input_order), end(m_input_order), i);
            OP_VALIDATION(this, it != end(m_input_order))
                << "Input axis order is not a permutation of argument's axis indices (axis order: "
                << m_input_order << ", argument shape: " << input_shape << ").";
        }

        // TODO(amprocte): make a partial_shape_size() analogous to shape_size().
        nnfusion::Dimension input_shape_product = 1;
        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            input_shape_product *= input_shape[i];
        }

        if (input_shape_product.is_static())
        {
            OP_VALIDATION(this, size_t(input_shape_product) == nnfusion::shape_size(m_output_shape))
                << "Product of output shape dimensions does not match product of argument shape "
                   "dimensions "
                << "(output shape: " << m_output_shape << ", argument shape: " << input_shape
                << ").";
        }
    }

    if (!std::is_sorted(m_input_order.begin(), m_input_order.end()))
    {
        m_is_transpose = true;
    }
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), m_output_shape);
}