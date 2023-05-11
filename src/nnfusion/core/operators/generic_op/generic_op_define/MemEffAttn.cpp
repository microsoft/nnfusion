// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MemEffAttn)
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
        // q: [b, h, q, d], k & v : [b, h, k, d]
        // lse: [b, h, q], m: [b, h, q], acco: [b,h, q, d]
        NNFUSION_CHECK(gnode->get_in_edges().size() == 6);

        auto in_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(in_shape.size() == 4);
        nnfusion::Shape outshape{in_shape[0], in_shape[1], in_shape[2], in_shape[3]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string { return ""; });
REGISTER_OP(MemEffAttnBasic)
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

    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t batch_size = generic_op->localOpConfig.getRoot()["batch_size"];
        size_t num_heads = generic_op->localOpConfig.getRoot()["num_heads"];
        size_t seq_len = generic_op->localOpConfig.getRoot()["seq_len"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];
        size_t head_size = generic_op->localOpConfig.getRoot()["head_size"];

        nnfusion::Shape output_shape;
        if (stage == 0 || stage == 2)
        {
            output_shape = {batch_size, num_heads, seq_len, block_size};
        }
        else if (stage == 3 || stage == 5)
        {
            output_shape = {batch_size, num_heads, seq_len, head_size};
        }
        else
        {
            output_shape = {batch_size, num_heads, seq_len};
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
        bool s = stage == 0;
        string expression_template;
        if (stage == 0)
        {
            //q, k -> qk
            expression_template =
                R"( @output0@[B, H, Q, Kc] +=! @input0@[B, H, Q, D] * @input1@[B, H, Kc, D];  )";
        }
        else if (stage == 1)
        {
            // qk -> max
            //max, lse_i -> m_ij
            //m_ij->m_i

            // qk, lse_i -> m_ij
            expression_template =
                R"( mediate0[B, H, Q] >=! @input0@[B, H, Q, Kc]; @output0@[B, H, Q] = (mediate0[B, H, Q] * const(@softmax_scale@).cast(input0[0].dtype())).call(`max`, [@input1@[B, H, Q]]);)";
        }
        else if (stage == 2)
        {
            //m_ij, qk -> p
            expression_template =
                R"( mediate0[B, H, Q, Kc] = @input0@[B, H, Q] where Kc in @block_size@; @output0@[B, H, Q, Kc] = (@input1@[B, H, Q, Kc] * const(@softmax_scale@).cast(input0[0].dtype()) - mediate0[B, H, Q, Kc]).call(`exp`);)";
        }
        else if (stage == 3)
        {
            //m_i, m_ij -> acc_o_scale
            // acc_o_scale, acc_o, p, v -> acc_o

            // m_i, m_ij, acc_o, p, v->acc_o
            expression_template =
                R"( mediate0[B, H, Q] = (@input0@[B, H, Q] - @input1@[B, H, Q]).call(`exp`); mediate1[B, H, Q, D] = mediate0[B, H, Q] where D in @head_size@; mediate2[B, H, Q, D] = @input2@[B, H, Q, D] * mediate1[B, H, Q, D]; mediate3[B, H, Q, D] +=! @input3@[B, H, Q, Kc] * @input4@[B, H, Kc, D]; @output0@[B, H, Q, D] = mediate2[B, H, Q, D] + mediate3[B, H, Q, D];)";
        }
        else if (stage == 4)
        {
            //p->l_ij
            // m_ij, lse_i, l_ij -> lse_i
            // m_ij, lse_i, p-> lse_i
            expression_template =
                R"( mediate0[B, H, Q] +=! @input2@[B, H, Q, Kc]; @output0@[B, H, Q] = @input0@[B, H, Q] + ((@input1@[B, H, Q] - @input0@[B, H, Q]).call(`exp`) + mediate0[B, H, Q]).call(`log`); )";
        }
        else if (stage == 5)
        {
            // m_i, lse_i, acc_o-> out
            expression_template =
                R"( mediate0[B, H, Q, D] = (@input0@[B, H, Q] - @input1@[B, H, Q]).call(`exp`) where D in @head_size@; @output0@[B, H, Q, D] = @input2@[B, H, Q, D] * mediate0[B, H, Q, D];)";
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID: " << stage;
        }
        std::string expression_code =
            op::create_code_from_template(expression_template,
                                          {{"softmax_scale", softmax_scale},
                                           {"head_size", head_size},
                                           {"block_size", block_size}});

        if ((stage == 0 || stage == 3) &&
            curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            expression_code += "## @: tensorCoreConfig=(2, 3)";
        }
        return expression_code;
    });