// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

std::string reduce_template(std::shared_ptr<graph::GNode> curr, std::string reduce_op = "+=!")
{
    auto make_layout = [](const std::set<int>& axes) -> std::string {
        std::string ret = "";
        for (auto ax : axes)
            ret += ", N" + std::to_string(ax);
        return "[" + (axes.empty() ? "N" : ret.substr(2)) + "]";
    };
    auto attrs = curr->get_op_ptr()->serialize();

    auto input_shape = curr->get_input_shape(0);
    std::vector<int> _axes = attrs["reduction_axes"];
    auto axes = std::set<int>(_axes.begin(), _axes.end());

    std::set<int> input_ax, output_ax;
    size_t reduce_size = 1L;
    for (int i = 0; i < input_shape.size(); ++i)
    {
        if (!axes.count(i))
            output_ax.insert(i);
        else
            reduce_size *= input_shape[i];
        input_ax.insert(i);
    }

    auto expression = "@output0@" + make_layout(output_ax) + " " + reduce_op + " @input0@" +
                      make_layout(input_ax);
    if (output_ax.empty())
        expression += " where N in 1;";
    else
        expression += ";";

    // FIXME: Need to include annotation
    if (reduce_size == 1L)
        expression += " ## @: memcpy";

    return expression;
}

REGISTER_OP(Sum)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        return reduce_template(curr, "+=!");
    });

REGISTER_OP(Max)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        return reduce_template(curr, ">=!");
    });

REGISTER_OP(Min)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        return reduce_template(curr, "<=!");
    });

REGISTER_OP(Product)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        return reduce_template(curr, "*=!");
    });