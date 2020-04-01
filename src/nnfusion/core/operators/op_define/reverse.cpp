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