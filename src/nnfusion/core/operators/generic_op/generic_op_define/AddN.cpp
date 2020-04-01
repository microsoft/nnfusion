// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(AddN).attr<nnfusion::op::OpConfig::any>("T").infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        // enforce is like assert, but when thing goes wrong, it will print error message.
        CHECK(gnode->get_input_size() >= 2) << "Inputs of AddN operator should not be less than 2.";

        auto& shape_0 = gnode->get_input_shape(0);
        for (int i = 1; i < gnode->get_input_size(); i++)
        {
            auto& shape_n = gnode->get_input_shape(i);
            CHECK(shape_0.size() == shape_n.size()) << "Shape dimension size not match.";
            for (int j = 0; j < shape_0.size(); j++)
            {
                CHECK(shape_0[j] == shape_n[j]) << "Dimension " << j << " in shapes must be equal.";
            }
        }

        nnfusion::Shape output_shape_0(shape_0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
