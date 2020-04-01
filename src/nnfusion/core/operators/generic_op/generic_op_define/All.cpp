// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(All)
    .attr<int>("axis", -1)
    .attr<bool>("keep_dims", false)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(1 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool keep_dims = generic_op->localOpConfig.getRoot()["keep_dims"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        if (axis == -1)
        {
            axis = shape_0.size() - 1;
        }

        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        if (keep_dims)
            output_shape_0.push_back(1);
        for (int i = axis + 1; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
