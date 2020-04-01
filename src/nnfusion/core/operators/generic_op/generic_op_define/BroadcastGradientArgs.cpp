// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(BroadcastGradientArgs).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    nnfusion::Shape output_shape = {};
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    gnode->set_output_type_and_shape(1, gnode->get_input_element_type(0), output_shape);
});
