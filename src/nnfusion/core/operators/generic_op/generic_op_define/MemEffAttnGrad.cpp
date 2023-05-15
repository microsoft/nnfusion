// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MemEffAttnGradPre)
    .attr<size_t>("batch_size")
    .attr<size_t>("num_heads")
    .attr<size_t>("seq_len")
    .attr<size_t>("seq_len_kv")
    .attr<size_t>("head_size")
    .attr<size_t>("head_size_v")
    .attr<float>("softmax_scale", 0.1580810546875)
    .attr<float>("p_dropout", 0)
    .attr<bool>("is_causal", false) // Whether every token can only attend to previous tokens
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // o: [b,h, q, d], do: [b,h, q, d]
        // out: delta:[b, h, q]

        NNFUSION_CHECK(gnode->get_in_edges().size() == 2);

        auto in_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(in_shape.size() == 4);
        nnfusion::Shape outshape{in_shape[0], in_shape[1], in_shape[2]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string { 
        std::string expression_code =
                R"( @output0@[B, H, Q] +=! @input0@[B, H, Q, D] * @input1@[B, H, Q, D];)";
        return expression_code;
    });
REGISTER_OP(MemEffAttnGrad)
    .attr<size_t>("batch_size")
    .attr<size_t>("num_heads")
    .attr<size_t>("seq_len")
    .attr<size_t>("seq_len_kv")
    .attr<size_t>("head_size")
    .attr<size_t>("head_size_v")
    .attr<float>("softmax_scale", 0.1580810546875)
    .attr<float>("p_dropout", 0)
    .attr<bool>("is_causal", false) // Whether every token can only attend to previous tokens
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // q: [b, h, q, d], k & v : [b, h, k, d], do: [b,h, q, d], 
        //lse:[b,h,q], delta:[b, h, q]
        // dq_accum: [b, h, q, d]
        // out: dq, dk, dv

        NNFUSION_CHECK(gnode->get_in_edges().size() == 9);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
        gnode->set_output_type_and_shape(1, gnode->get_input_element_type(1), gnode->get_input_shape(1));
        gnode->set_output_type_and_shape(2, gnode->get_input_element_type(2), gnode->get_input_shape(2));
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
    return ""; });

REGISTER_OP(MemEffAttnGradBasic)
    .attr<size_t>("batch_size")
    .attr<size_t>("num_heads")
    .attr<size_t>("seq_len")
    .attr<size_t>("seq_len_kv")
    .attr<size_t>("head_size")
    .attr<float>("p_dropout", 0)
    .attr<int>("stage")
    .attr<float>("softmax_scale", 0.125)
    .attr<bool>("is_causal", false) // Whether every token can only attend to previous tokens
    .attr<size_t>("block_size", 128)
    .attr<int>("atomic_add", false)

    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t batch_size = generic_op->localOpConfig.getRoot()["batch_size"];
        size_t num_heads = generic_op->localOpConfig.getRoot()["num_heads"];
        size_t seq_len = generic_op->localOpConfig.getRoot()["seq_len"];
        size_t seq_len_kv = generic_op->localOpConfig.getRoot()["seq_len_kv"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];
        size_t head_size = generic_op->localOpConfig.getRoot()["head_size"];

        nnfusion::Shape output_shape;
        if (stage == 0 || stage == 1 || stage == 3 || stage == 4)
        {
            output_shape = {batch_size, num_heads, seq_len, seq_len_kv};
        }
        else if (stage == 2 || stage == 5)
        {
            output_shape = {batch_size, num_heads, seq_len_kv, head_size};
        }
        else
        {
            output_shape = {batch_size, num_heads, seq_len, head_size};
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        // float softmax_scale = generic_op->localOpConfig.getRoot()["softmax_scale"];
        size_t num_heads = generic_op->localOpConfig.getRoot()["num_heads"];
        float softmax_scale = 1.0 / num_heads;
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        int block_size = generic_op->localOpConfig.getRoot()["block_size"];
        int head_size = generic_op->localOpConfig.getRoot()["head_size"];
        bool atomic_add = generic_op->localOpConfig.getRoot()["atomic_add"];
        string expression_template;
        if (stage == 0)
        {
            //q, k -> qk 
            expression_template =
                R"( @output0@[B, H, Q, K] +=! @input0@[B, H, Q, D] * @input1@[B, H, K, D];  )";
        }
        else if (stage == 1)
        {
            // qk, lse_i -> p
            expression_template =
                R"(@output0@[B, H, Q, K] = (@input0@[B, H, Q, K] * const(@softmax_scale@).cast(input0[0].dtype()) - @input1@[B, H, Q]).call(`exp`);)";
        }
        else if (stage == 2)
        {
            //p, do, dv -> dv
            expression_template =
                R"( mediate0[B, H, K, D] +=! @input0@[B, H, Q, K] * @input1@[B, H, Q, D]; @output0@[B, H, K, D] = mediate0[B, H, K, D] + @input2@[B, H, K, D];)";
        }
        else if (stage == 3)
        {
            //do, v-> dp
            expression_template =
                R"(@output0@[B, H, Q, K] +=! @input0@[B, H, Q, D] * @input1@[B, H, K, D];)";
        }
        else if (stage == 4)
        {
            // p, dp, delta -> ds
            expression_template =
                R"(@output0@[B, H, Q, K] = @input0@[B, H, Q, K] * (@input1@[B, H, Q, K] - @input2@[B, H, Q]) * const(@softmax_scale@).cast(input0[0].dtype());)";
        }
        else if (stage == 5)
        {
            // ds, q, dk -> dk
            expression_template =
                R"(mediate0[B, H, K, D] +=! @input0@[B, H, Q, K] * @input1@[B, H, Q, D]; @output0@[B, H, K, D] = mediate0[B, H, K, D] + @input2@[B, H, K, D]; )";
        }
        else if (stage == 6)
        {
            if (atomic_add)
            {
                // ds, k -> dq
                expression_template =
                R"(@output0@[B, H, Q, D] +=! @input0@[B, H, Q, K] * @input1@[B, H, K, D];)";
            }
            else
            {
                // ds, k, dq -> dq
                expression_template =
                R"(mediate0[B, H, Q, D] +=! @input0@[B, H, Q, K] * @input1@[B, H, K, D]; @output0@[B, H, Q, D] = mediate0[B, H, Q, D] + @input2@[B, H, Q, D];)";
            }
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID: " << stage;
        }
        std::string expression_code =
            op::create_code_from_template(expression_template,
                                          {
                                            {"softmax_scale", softmax_scale},
                                          });

        if ((stage != 1 && stage != 4) &&
            curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            expression_code += "## @: tensorCoreConfig=(2, 3)";
        }
        return expression_code;
    });