// Microsoft (c) 2019, NNFusion Team

#include "select.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Select::Select()
    : Op("Select")
{
}

void Select::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this,
                  gnode->get_input_element_type(0).is_dynamic() ||
                      gnode->get_input_element_type(0) == element::boolean)
        << "Argument 0 does not have boolean element type (element type: "
        << gnode->get_input_element_type(0) << ").";

    nnfusion::PartialShape result_shape = gnode->get_input_partial_shape(0);

    OP_VALIDATION(
        this, nnfusion::PartialShape::merge_into(result_shape, gnode->get_input_partial_shape(1)))
        << "Argument shapes are inconsistent.";
    OP_VALIDATION(
        this, nnfusion::PartialShape::merge_into(result_shape, gnode->get_input_partial_shape(2)))
        << "Argument shapes are inconsistent.";

    nnfusion::element::Type result_et;

    OP_VALIDATION(this,
                  nnfusion::element::Type::merge(result_et,
                                                 gnode->get_input_element_type(1),
                                                 gnode->get_input_element_type(2)))
        << "Argument 1 and 2 element types are inconsistent.";

    gnode->set_output_type_and_shape(0, result_et, result_shape);
}