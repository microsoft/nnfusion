// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(ftc_rewrite);

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
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape_0;
        auto sym_out = std::make_shared<SymShape>();

        auto sym_a = input_shape_0.get_sym_shape()
                         ? input_shape_0.get_sym_shape()
                         : std::make_shared<nnfusion::SymShape>(input_shape_0);
        auto sym_b = input_shape_1.get_sym_shape()
                         ? input_shape_1.get_sym_shape()
                         : std::make_shared<nnfusion::SymShape>(input_shape_1);
        bool has_sym_input = (input_shape_0.get_sym_shape() || input_shape_1.get_sym_shape());

        NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
        NNFUSION_CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));

        for (int i = 0; i < input_shape_0.size() - 2; i++)
        {
            NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
            sym_out->push_back(sym_a->at(i));
        }

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];
        int m1 = input_shape_1[input_shape_1.size() - 2],
            n1 = input_shape_1[input_shape_1.size() - 1];

        SymDim sm0 = (*sym_a)[input_shape_0.size() - 2], sn0 = (*sym_a)[input_shape_0.size() - 1];
        SymDim sm1 = (*sym_b)[input_shape_1.size() - 2], sn1 = (*sym_b)[input_shape_1.size() - 1];

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        if (!trans_A && !trans_B)
            NNFUSION_CHECK(m1 == n0)
        , output_shape_0.push_back(m0), output_shape_0.push_back(n1), sym_out->push_back(sm0),
            sym_out->push_back(sn1);
        else if (!trans_A && trans_B) NNFUSION_CHECK(n0 == n1), output_shape_0.push_back(m0),
            output_shape_0.push_back(m1), sym_out->push_back(sm0), sym_out->push_back(sm1);
        else if (trans_A && !trans_B) NNFUSION_CHECK(m0 == m1), output_shape_0.push_back(n0),
            output_shape_0.push_back(n1), sym_out->push_back(sn0), sym_out->push_back(sn1);
        else // trans_A && trans_B
            NNFUSION_CHECK(m0 == n1),
            output_shape_0.push_back(n0), output_shape_0.push_back(m1), sym_out->push_back(sn0),
            sym_out->push_back(sm1);
        if (has_sym_input)
            output_shape_0.set_sym_shape(sym_out);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(gnode->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        nnfusion::Shape output_shape = gnode->get_output_shape(0);

        NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
        NNFUSION_CHECK(gnode->get_input_element_type(0) == gnode->get_input_element_type(1));
        // NNFUSION_CHECK(input_shape_0[0] == input_shape_1[0] || input_shape_0[0] == 1 ||
        //                input_shape_1[0] == 1);

        int m0 = input_shape_0[input_shape_0.size() - 2],
            n0 = input_shape_0[input_shape_0.size() - 1];

        int batch = 1;
        for (int i = 0; i < input_shape_0.size() - 2; ++i)
        {
            NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
            batch *= input_shape_0[i];
        }

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        int N = trans_A ? input_shape_0[input_shape_0.size() - 1]
                        : input_shape_0[input_shape_0.size() - 2];
        int K = trans_A ? input_shape_0[input_shape_0.size() - 2]
                        : input_shape_0[input_shape_0.size() - 1];
        int M = trans_B ? input_shape_1[input_shape_1.size() - 2]
                        : input_shape_1[input_shape_1.size() - 1];

        std::string input0 = trans_A ? "args(\"input0\")[b, k, n]" : "args(\"input0\")[b, n, k]";
        std::string input1 = trans_B ? "args(\"input1\")[b, m, k]" : "args(\"input1\")[b, k, m]";

        std::vector<int> shape0 = {batch, N, K};
        if (trans_A)
            std::swap(shape0[1], shape0[2]);
        std::vector<int> shape1 = {batch, K, M};
        if (trans_B)
            std::swap(shape1[1], shape1[2]);
        std::vector<int> outshape = {batch, N, M};

        auto expression = op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@k@); output(@output_shape@, lambda b, n, m: tvm.sum(@input0_expr@ * @input1_expr@, axis=k));  ## @: plan/batch_matmul_v1)",
            {{"input_shape_0", vector_to_string(shape0)},
             {"input_shape_1", vector_to_string(shape1)},
             {"output_shape", vector_to_string(outshape)},
             {"k", K},
             {"input0_expr", input0},
             {"input1_expr", input1}});

        return expression;
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        NNFUSION_CHECK(curr->get_input_size() == 2);

        const nnfusion::Shape& input_shape_0 = curr->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = curr->get_input_shape(1);
        nnfusion::Shape output_shape_0 = curr->get_output_shape(0);

        NNFUSION_CHECK(input_shape_0.size() == input_shape_1.size());
        NNFUSION_CHECK(curr->get_input_element_type(0) == curr->get_input_element_type(1));

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
        bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@ * @input1@@input1_layout@; )";

        std::vector<std::string> output0_layout;
        std::vector<std::string> input0_layout;
        std::vector<std::string> input1_layout;

        for (size_t i = 0; i < output_shape_0.size() - 2; ++i)
        {
            std::string batch_dim = "B" + to_string(i);
            output0_layout.push_back(batch_dim);
            input0_layout.push_back(batch_dim);
            input1_layout.push_back(batch_dim);
        }

        output0_layout.push_back("N");
        output0_layout.push_back("M");

        if (trans_A)
        {
            input0_layout.push_back("K");
            input0_layout.push_back("N");
        }
        else
        {
            input0_layout.push_back("N");
            input0_layout.push_back("K");
        }

        if (trans_B)
        {
            input1_layout.push_back("M");
            input1_layout.push_back("K");
        }
        else
        {
            input1_layout.push_back("K");
            input1_layout.push_back("M");
        }

        op::OpConfig::any op_config;
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(input0_layout);
        op_config["input1_layout"] = vector_to_string<std::vector<std::string>>(input1_layout);
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_layout);

        auto ir = op::create_code_from_template(ir_template, op_config);
        if (FLAGS_ftc_rewrite && curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            ir += "## @: tensorCoreConfig=(" + to_string(output0_layout.size() - 2) + ", " +
                  to_string(output0_layout.size() - 1) + ")";
        }
        return ir;
    });
