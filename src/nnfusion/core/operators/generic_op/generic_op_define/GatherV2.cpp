// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherV2).attr<int>("axis", 0).infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(gnode->get_input_size() == 2);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        int axis = generic_op->localOpConfig.getRoot()["axis"];

        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
            output_shape_0.push_back(input_shape_0[i]);
        for (int i = 0; i < input_shape_1.size(); ++i)
            output_shape_0.push_back(input_shape_1[i]);
        for (int i = axis + 1; i < input_shape_0.size(); ++i)
            output_shape_0.push_back(input_shape_0[i]);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
