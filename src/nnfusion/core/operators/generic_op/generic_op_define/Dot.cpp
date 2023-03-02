// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(ftc_rewrite);

REGISTER_OP(Dot)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto product = [&](const nnfusion::Shape& shape, int start, int stop) {
            if (start < 0)
                start += (int)shape.size();
            if (stop <= 0)
                stop += (int)shape.size();
            size_t base = 1;
            for (int i = start; i < stop; ++i)
                base *= shape[i];
            return base;
        };

        std::vector<size_t> shape_0 = {
            product(curr->get_input_shape(0), 0, curr->get_input_shape(0).size() - 1),
            curr->get_input_shape(0).back()};
        std::vector<size_t> shape_1 = {
            product(curr->get_input_shape(1), 0, curr->get_input_shape(1).size() - 1),
            curr->get_input_shape(1).back()};
        std::vector<size_t> shape_out = {
            product(curr->get_output_shape(0), 0, curr->get_output_shape(0).size() - 1),
            curr->get_output_shape(0).back()};

        int N = _op->get_transpose_A() ? shape_0[1] : shape_0[0];
        int K = _op->get_transpose_A() ? shape_0[0] : shape_0[1];
        int M = _op->get_transpose_B() ? shape_1[0] : shape_1[1];

        return op::create_code_from_template(
            R"( - input("input0", @shape_0@); input("input1", @shape_1@); k = loop(@K@); output(@output_shape@, lambda i, j: tvm.sum(args("input0")[@dimA@] * args("input1")[@dimB@], axis=k));  ## @: plan/matmul_v1)",
            {{"shape_0", vector_to_string(shape_0)},
             {"shape_1", vector_to_string(shape_1)},
             {"dimA", _op->get_transpose_A() ? "k,i" : "i,k"},
             {"dimB", _op->get_transpose_B() ? "j,k" : "k,j"},
             {"K", K},
             {"output_shape", vector_to_string(shape_out)}});
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        auto _op = static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        NNFUSION_CHECK(_op->get_reduction_axes_count() == 1);
        auto input0_shape = curr->get_input_shape(0);
        auto input1_shape = curr->get_input_shape(1);
        NNFUSION_CHECK(input0_shape.size() >= 2 && input1_shape.size() >= 2);

        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@ * @input1@@input1_layout@; )";

        vector<string> input0_layout, input1_layout, output0_layout;
        for (size_t i = 0; i + 2 < input0_shape.size(); i++)
        {
            input0_layout.push_back("S" + std::to_string(i));
            output0_layout.push_back("S" + std::to_string(i));
        }
        output0_layout.push_back("N");
        output0_layout.push_back("M");
        input0_layout.push_back(_op->get_transpose_A() ? "K" : "N");
        input0_layout.push_back(_op->get_transpose_A() ? "N" : "K");
        input1_layout.push_back(_op->get_transpose_B() ? "M" : "K");
        input1_layout.push_back(_op->get_transpose_B() ? "K" : "M");
        for (size_t i = 0; i + 2 < input1_shape.size(); i++)
        {
            input1_layout.push_back("E" + std::to_string(i));
            output0_layout.push_back("E" + std::to_string(i));
        }

        op::OpConfig::any op_config;
        op_config["input0_layout"] = nnfusion::vector_to_string(input0_layout);
        op_config["input1_layout"] = nnfusion::vector_to_string(input1_layout);
        op_config["output0_layout"] = nnfusion::vector_to_string(output0_layout);
        auto ir = op::create_code_from_template(ir_template, op_config);
        if (FLAGS_ftc_rewrite && curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            ir += "## @: tensorCoreConfig=(" + to_string(output0_layout.size() - 2) + ", " +
                  to_string(output0_layout.size() - 1) + ")";
        }

        return ir;
    });
