// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Range).attr<int>("start").attr<int>("limit").attr<int>("delta").infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        float start = generic_op->localOpConfig.getRoot()["start"];
        float limit = generic_op->localOpConfig.getRoot()["limit"];
        float delta = generic_op->localOpConfig.getRoot()["delta"];
        int num = (int)((limit - start + delta - 1) / delta);

        nnfusion::Shape output_shape_0;
        output_shape_0.push_back(num);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
