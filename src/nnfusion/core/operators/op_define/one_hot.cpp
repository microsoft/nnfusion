// Microsoft (c) 2019, NNFusion Team

#include "one_hot.hpp"
#include "nnfusion/core/graph/gnode.hpp"
using namespace std;
using namespace nnfusion::op;

OneHot::OneHot(const nnfusion::PartialShape& shape, size_t one_hot_axis)
    : Op("OneHot")
    , m_shape(shape)
    , m_one_hot_axis(one_hot_axis)
{
}

void OneHot::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    nnfusion::element::Type arg_et = gnode->get_input_element_type(0);
    nnfusion::PartialShape arg_shape = gnode->get_input_partial_shape(0);
    nnfusion::Rank arg_rank = arg_shape.rank();

    OP_VALIDATION(this, m_shape.rank().is_static()) << "Requested result shape has dynamic rank.";

    OP_VALIDATION(this, m_one_hot_axis < static_cast<size_t>(m_shape.rank()))
        << "One-hot axis (" << m_one_hot_axis
        << ") is out of bounds (requested result shape: " << m_shape << ").";

    OP_VALIDATION(this, m_shape[m_one_hot_axis].is_static())
        << "Requested result shape (" << m_shape << ") has dynamic dimension at the one-hot axis "
        << "(" << m_one_hot_axis << ").";

    nnfusion::PartialShape result_shape{m_shape};

    if (arg_rank.is_static())
    {
        std::vector<nnfusion::Dimension> expected_input_dims(static_cast<size_t>(m_shape.rank()));
        for (size_t i = 0; i < static_cast<size_t>(m_shape.rank()); i++)
        {
            expected_input_dims[i] = m_shape[i];
        }
        expected_input_dims.erase(expected_input_dims.begin() + m_one_hot_axis);
        nnfusion::PartialShape expected_input_shape{expected_input_dims};

        nnfusion::PartialShape merged_input_shape{expected_input_shape};
        OP_VALIDATION(this, nnfusion::PartialShape::merge_into(merged_input_shape, arg_shape))
            << "Argument shape " << arg_shape << " does not match the expected shape of "
            << expected_input_shape << ".";

        std::vector<nnfusion::Dimension> output_dims(
            static_cast<size_t>(merged_input_shape.rank()));
        for (size_t i = 0; i < static_cast<size_t>(merged_input_shape.rank()); i++)
        {
            output_dims[i] = merged_input_shape[i];
        }
        output_dims.insert(output_dims.begin() + m_one_hot_axis, m_shape[m_one_hot_axis]);
        result_shape = nnfusion::PartialShape{output_dims};
    }

    gnode->set_output_type_and_shape(0, arg_et, result_shape);
}