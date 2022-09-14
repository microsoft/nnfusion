#include "h2d.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

H2D::H2D()
    : Op("h2d")
{
}

void H2D::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto element_type = gnode->get_input_element_type(0);
    gnode->set_output_type_and_shape(0, element_type, gnode->get_input_partial_shape(0));
}