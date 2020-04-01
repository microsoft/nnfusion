// Microsoft (c) 2019, NNFusion Team

#include "convert.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

Convert::Convert(const nnfusion::element::Type& element_type)
    : Op("Convert")
    , m_element_type(element_type)
{
}

void Convert::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    gnode->set_output_type_and_shape(0, m_element_type, gnode->get_input_shape(0));
}
