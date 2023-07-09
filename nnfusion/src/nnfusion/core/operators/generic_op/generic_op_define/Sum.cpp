// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Sum)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
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

        auto expression =
            "@output0@" + make_layout(output_ax) + " +=! @input0@" + make_layout(input_ax);
        if (output_ax.empty())
            expression += " where N in 1";

        // FIXME: Need to include annotation
        if (reduce_size == 1L)
            expression += " ## @: memcpy";

        return expression;
    });
/*
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Sum>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        auto axes = _op->get_reduction_axes();
        auto in_shape = curr->get_input_shape(0);

        std::string dtype;
        NNFUSION_CHECK(
            element::Type::nnfusion_element_type_to_dtype_string(curr->get_element_type(), dtype));
        NNFUSION_CHECK(dtype == "float32") << "Unsupported Sum dtype " << dtype;

        std::vector<int> ordered_axes(axes.begin(), axes.end());
        std::sort(ordered_axes.begin(), ordered_axes.end());

        size_t num_elements = 1, sample = 1;
        for (int i = 0; i < in_shape.size(); ++i)
            num_elements *= in_shape[i];
        for (auto it : axes)
            sample *= in_shape[it];
        assert(sample != 0);

        // ReduceHigh
        if (axes.size() && ordered_axes.front() == 0 &&
            ordered_axes.back() == ordered_axes.size() - 1)
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[C] +=! input0[N, C]", input_dict={"input0": {"dtype": "@dtype@", "shape": [@sample@, @batch@]}});  ## :@ plan/reduce_sum_v1)",
                {
                    {"dtype", dtype}, {"sample", sample}, {"batch", num_elements / sample},
                });
        }

        // ReduceLow
        if (axes.size() && ordered_axes.front() == in_shape.size() - ordered_axes.size() &&
            ordered_axes.back() == in_shape.size() - 1)
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "@dtype@", "shape": [@batch@, @sample@]}});  ## :@ plan/reduce_sum_v1)",
                {
                    {"dtype", dtype}, {"sample", sample}, {"batch", num_elements / sample},
                });
        }

        // ReduceNone
        if (!ordered_axes.size())
        {
            return op::create_code_from_template(
                R"( - einstein_v2("output0[N] = input0[N]", input_dict={"input0": {"dtype": "@dtype@", "shape": [@num_elements@]}});  ## @annotation: memcpy)",
                {
                    {"dtype", dtype}, {"num_elements", num_elements},
                });
        }
        return "";

        auto input_shape = curr->get_input_shape(0);

        int min_axis = axes.size() + 1;
        if (axes.size() == 0)
        {
            min_axis = 0;
        }
        else
        {
            for (auto& axis : axes)
                min_axis = min(min_axis, (int)axis);
        }

        if (input_shape.size() - axes.size() == min_axis || axes.size() == 0)
        {
            int batch = 1, sample = 1;
            for (int i = 0; i < min_axis; ++i)
                batch *= input_shape[i];
            for (int i = min_axis; i < input_shape.size(); ++i)
                sample *= input_shape[i];

            return op::create_code_from_template(
                " - input(\"input0\", [@batch@, @sample@]); output([@batch@], "
                "topi=topi.sum(args(\"input0\"), axis=@axis@, keepdims=True));",
                {{"batch", batch}, {"sample", sample}, {"axis", axes.size() != 0 ? "1" : "None"}});
        }
        else
        {
            return op::create_code_from_template(
                " - input(\"input0\", @input_shape@); output(@output_shape@, "
                "topi=topi.sum(args(\"input0\"), axis=@axis@, keepdims=True));",
                {{"input_shape", vector_to_string(input_shape)},
                 {"output_shape", vector_to_string(curr->get_output_shape(0))},
                 {"axis", vector_to_string(axes)}});
        }
    });
    */
