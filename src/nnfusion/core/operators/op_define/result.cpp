// Microsoft (c) 2019, NNFusion Team

#include "result.hpp"
#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;

Result::Result()
    : Op("Result")
{
}

void Result::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    OP_VALIDATION(this, gnode->get_input_size() == 1) << "Argument has " << gnode->get_input_size()
                                                      << " outputs (1 expected).";

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), gnode->get_input_partial_shape(0));
}
