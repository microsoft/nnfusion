// Microsoft (c) 2019, NNFusion Team

#include <algorithm>
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "nnfusion/core/graph/gnode.hpp"
#include "reverse_sequence.hpp"

using namespace std;
using namespace nnfusion::op;

ReverseSequence::ReverseSequence(size_t batch_axis, size_t seq_axis)
    : Op("ReverseSequence")
    , m_batch_axis(batch_axis)
    , m_seq_axis(seq_axis)
{
}

void ReverseSequence::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto input_shape = gnode->get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    OP_VALIDATION(this, input_rank.is_dynamic() || m_batch_axis < size_t(input_rank))
        << "Batch axis index (" << m_batch_axis
        << ") is out of bounds (argument shape: " << input_shape << ").";

    OP_VALIDATION(this, input_rank.is_dynamic() || m_seq_axis < size_t(input_rank))
        << "Sequence axis index (" << m_seq_axis
        << ") is out of bounds (argument shape: " << input_shape << ").";

    auto indices_shape = gnode->get_input_partial_shape(1);
    auto indices_rank = indices_shape.rank();

    OP_VALIDATION(this, indices_rank.is_dynamic() || size_t(indices_rank) == 1)
        << "Sequence indices must be a 1-dimensional tensor (sequence indices shape: "
        << gnode->get_input_partial_shape(1) << ").";

    PartialShape output_shape{input_shape};

    if (input_rank.is_static() && indices_rank.is_static())
    {
        Dimension merged_sequence_length;

        OP_VALIDATION(this,
                      nnfusion::Dimension::merge(
                          merged_sequence_length, input_shape[m_batch_axis], indices_shape[0]))
            << "Sequence length (" << indices_shape[0] << ") is not equal to batch axis "
            << "dimension (" << input_shape[m_batch_axis] << ") (argument shape: " << input_shape
            << ", sequence indices shape: " << indices_shape << ").";
        output_shape[m_batch_axis] = merged_sequence_length;
    }

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
}
