// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"


static string make_layout(const std::set<int>& axes) {
    std::string ret = "";
    for (auto ax : axes)
        ret += ", N" + std::to_string(ax);
    return "[" + (axes.empty() ? "N" : ret.substr(2)) + "]";
};

REGISTER_OP(Softmax)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        auto op = static_pointer_cast<nnfusion::op::Softmax>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto input_shape = curr->get_input_shape(0);
        auto axes = op->get_axes();

        std::set<int> input_ax, output_ax;
        for (int i = 0; i < input_shape.size(); ++i)
        {
            if (!axes.count(i))
                output_ax.insert(i);
            input_ax.insert(i);
        }

        auto expression_template =
            R"( mediate0@temp_layout@ >=! input0@input0_layout@; mediate1@temp_layout@ +=! (input0@input0_layout@ - mediate0@temp_layout@).call(`exp`); output0@input0_layout@ = (input0@input0_layout@  - mediate0@temp_layout@).call(`exp`) / mediate1@temp_layout@; )";

        std::string expression_code = op::create_code_from_template(
            expression_template,
            {{"temp_layout", make_layout(output_ax)}, {"input0_layout", make_layout(input_ax)}});
        return expression_code;
    });

REGISTER_OP(SoftmaxBasic)
    .attr<vector<int>>("axes")
    .attr<int>("stage")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        vector<int> axes = generic_op->localOpConfig.getRoot()["axes"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        nnfusion::Shape output_shape;
        if (stage == 1 || stage == 3) {
            output_shape = shape_0;
        } else {
            set<int> ax_set(axes.begin(), axes.end());
            for (int i = 0; i < shape_0.size(); i++) {
                if (ax_set.count(i)) continue;
                output_shape.push_back(shape_0[i]);
            }
        }
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        std::set<int> input_ax, output_ax;
        auto input_shape = curr->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        vector<int> axes = generic_op->localOpConfig.getRoot()["axes"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        set<int> ax_set(axes.begin(), axes.end());
        for (int i = 0; i < input_shape.size(); ++i)
        {
             if (!ax_set.count(i))
                output_ax.insert(i);
            input_ax.insert(i);
        }
        string expression_template;
        if (stage == 0) {
            expression_template =
                R"( @output0@@temp_layout@ >=! @input0@@input0_layout@; )";
        } else if (stage == 1) {
            expression_template =
                R"( @output0@@input0_layout@ = (@input0@@input0_layout@ - @input1@@temp_layout@).call(`exp`); )";
        } else if (stage == 2) {
            expression_template =
                R"( @output0@@temp_layout@ +=! @input0@@input0_layout@; )";
        } else if (stage == 3) {
            expression_template =
                R"( @output0@@input0_layout@ = @input0@@input0_layout@ / @input1@@temp_layout@; )";
        } else {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID.";
        }
        std::string expression_code = op::create_code_from_template(
            expression_template,
            {{"temp_layout", make_layout(output_ax)}, {"input0_layout", make_layout(input_ax)}});
        return expression_code;
    });
