// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

static string make_layout(const std::set<int>& axes)
{
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
