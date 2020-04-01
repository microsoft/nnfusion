// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(BatchMatMul)
    .attr<nnfusion::op::OpConfig::any>("adj_x", {{"b", false}})
    .attr<nnfusion::op::OpConfig::any>("adj_y", {{"b", false}})
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        if (!config["adj_x"]["b"].is_boolean())
            return false;
        if (!config["adj_y"]["b"].is_boolean())
            return false;
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape_0;

        CHECK(input_shape_0.size() == input_shape_1.size());
        CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));

        for (int i = 0; i < input_shape_0.size() - 2; i++)
        {
            CHECK(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
        }

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];
        int m1 = input_shape_1[input_shape_1.size() - 2],
            n1 = input_shape_1[input_shape_1.size() - 1];

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        if (!trans_A && !trans_B)
            CHECK(m1 == n0), output_shape_0.push_back(m0), output_shape_0.push_back(n1);
        else if (!trans_A && trans_B)
            CHECK(n0 == n1), output_shape_0.push_back(m0), output_shape_0.push_back(m1);
        else if (trans_A && !trans_B)
            CHECK(m0 == m1), output_shape_0.push_back(n0), output_shape_0.push_back(n1);
        else // trans_A && trans_B
            CHECK(m0 == n1), output_shape_0.push_back(n0), output_shape_0.push_back(m1);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
