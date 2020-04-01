// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Add).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    CHECK(2 == gnode->get_input_size());
    auto& shape_0 = gnode->get_input_shape(0);
    auto& shape_1 = gnode->get_input_shape(1);
    CHECK(shape_0.size() == shape_1.size());
    nnfusion::Shape output_shape_0;
    for (int i = 0; i < shape_0.size(); ++i)
    {
        if (shape_0[i] != shape_1[i])
            CHECK(shape_0[i] == 1 || shape_1[i] == 1);
        output_shape_0.push_back(std::max(shape_0[i], shape_1[i]));
    }
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
});
