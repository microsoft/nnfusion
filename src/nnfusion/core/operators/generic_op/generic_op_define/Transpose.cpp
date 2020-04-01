// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Transpose)
    .attr<std::vector<int>>("axes_order")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto& axes_order = generic_op->localOpConfig.getRoot()["axes_order"];
        CHECK(axes_order.size() == shape_0.size());
        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axes_order.size(); ++i)
            output_shape_0.push_back(shape_0[axes_order[i]]);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
