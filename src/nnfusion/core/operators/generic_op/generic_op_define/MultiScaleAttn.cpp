// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
// qr (B, NBL, NQ, BL, KD)
// kr (B, NBL, NQ, BL, KD)
// v (B, NBL, NQ, NV, BL, D)
// mask (NQ, NV, BLQ, BLK)
// cross_decay (NQ, NV)  (NQ, NV, 1, 1)
// inner_decay (NQ, NV, BL) (NQ, NV, BL, 1)

// kv_state (B, NQ, NV, KD, D)

// out: (B, NBL, NQ, NV, BL, D)
REGISTER_OP(MultiScaleAttn0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // qr (B, NBL, NQ, BLQ, KD)
        // kr (B, NBL, NQ, BLK, KD)
        // v (B, NBL, NQ, NV, BLK, D)
        // mask (NQ, NV, BLQ, BLK)

        // out(attn): (B, NBL, NQ, NV, BLQ, D)

        NNFUSION_CHECK(gnode->get_in_edges().size() == 4);

        auto qr_shape = gnode->get_input_shape(0);
        auto v_shape = gnode->get_input_shape(2);
        NNFUSION_CHECK(qr_shape.size() == 5);
        nnfusion::Shape outshape{
            qr_shape[0], qr_shape[1], qr_shape[2], v_shape[3], qr_shape[3], v_shape[5]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string { return ""; });

REGISTER_OP(MultiScaleAttn1)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // qr1 (B, NQ, BLQ, KD)
        // kr1 (B, NQ, BLK, KD)
        // v1 (B, NQ, NV, BLK, D)
        // mask (NQ, NV, BLQ, BLK)
        // cross_decay (NQ, NV)  (NQ, NV, 1, 1)
        // inner_decay (NQ, NV, BL) (NQ, NV, BLQ, 1)
        // ## kv_state (B, NQ, NV, KD, D)

        // out: (B, NQ, NV, BLQ, D)

        NNFUSION_CHECK(gnode->get_in_edges().size() == 6);

        auto qr_shape = gnode->get_input_shape(0);
        auto v_shape = gnode->get_input_shape(2);
        NNFUSION_CHECK(qr_shape.size() == 4);
        nnfusion::Shape outshape{qr_shape[0], qr_shape[1], v_shape[2], qr_shape[2], v_shape[4]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string { return ""; });
REGISTER_OP(MultiScaleAttnBasic0)
    .attr<int>("stage")
    .attr<size_t>("b")
    .attr<size_t>("nbl")
    .attr<size_t>("nq")
    .attr<size_t>("blq")
    .attr<size_t>("kd")
    .attr<size_t>("nv")
    .attr<size_t>("blk")
    .attr<size_t>("d")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t b = generic_op->localOpConfig.getRoot()["b"];
        size_t nbl = generic_op->localOpConfig.getRoot()["nbl"];
        size_t nq = generic_op->localOpConfig.getRoot()["nq"];
        size_t blq = generic_op->localOpConfig.getRoot()["blq"];
        size_t kd = generic_op->localOpConfig.getRoot()["kd"];
        size_t nv = generic_op->localOpConfig.getRoot()["nv"];
        size_t blk = generic_op->localOpConfig.getRoot()["blk"];
        size_t d = generic_op->localOpConfig.getRoot()["d"];

        int stage = generic_op->localOpConfig.getRoot()["stage"];

        nnfusion::Shape output_shape;
        if (stage == 0)
        {
            output_shape = {b, nbl, nq, blq, blk};
        }
        else if (stage == 1)
        {
            output_shape = {b, nbl, nq, nv, blq, blk};
        }
        else if (stage == 2)
        {
            output_shape = {b, nbl, nq, nv, blq, d};
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        int stage = generic_op->localOpConfig.getRoot()["stage"];
        string expression_template;
        if (stage == 0)
        {
            //qr, kr -> qk
            expression_template =
                R"(@output0@[B, NBL, NQ, BLQ, BLK] +=! @input0@[B, NBL, NQ, BLQ, KD] * @input1@[B, NBL, NQ, BLK, KD];)";
        }
        else if (stage == 1)
        {
            //qk, mask -> qkm
            expression_template =
                R"(@output0@[B, NBL, NQ, NV, BLQ, BLK] = @input0@[B, NBL, NQ, BLQ, BLK] * @input1@[NQ, NV, BLQ, BLK];)";
        }
        else if (stage == 2)
        {
            // qkm, v -> attn
            expression_template =
                R"(@output0@[B, NBL, NQ, NV, BLQ, D] +=! @input0@[B, NBL, NQ, NV, BLQ, BLK] * @input1@[B, NBL, NQ, NV, BLK, D];)";
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID: " << stage;
        }
        std::string expression_code = op::create_code_from_template(expression_template, {});

        if (curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            if (stage == 0)
                expression_code += "## @: tensorCoreConfig=(3, 4)";
            else if (stage == 2)
                expression_code += "## @: tensorCoreConfig=(4, 5)";
        }
        return expression_code;
    });

REGISTER_OP(MultiScaleAttnBasic1)
    .attr<int>("stage")
    .attr<size_t>("b")
    // .attr<size_t>("nbl")
    .attr<size_t>("nq")
    .attr<size_t>("blq")
    .attr<size_t>("kd")
    .attr<size_t>("nv")
    .attr<size_t>("blk")
    .attr<size_t>("d")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t b = generic_op->localOpConfig.getRoot()["b"];
        // size_t nbl = generic_op->localOpConfig.getRoot()["nbl"];
        size_t nq = generic_op->localOpConfig.getRoot()["nq"];
        size_t blq = generic_op->localOpConfig.getRoot()["blq"];
        size_t kd = generic_op->localOpConfig.getRoot()["kd"];
        size_t nv = generic_op->localOpConfig.getRoot()["nv"];
        size_t blk = generic_op->localOpConfig.getRoot()["blk"];
        size_t d = generic_op->localOpConfig.getRoot()["d"];

        int stage = generic_op->localOpConfig.getRoot()["stage"];

        nnfusion::Shape output_shape;
        if (stage == 0)
        {
            output_shape = {b, nq, nv, blk, d};
        }
        else if (stage == 1 || stage == 2)
        {
            output_shape = {b, nq, nv, kd, d};
        }
        else if (stage == 3)
        {
            output_shape = {b, nq, nv, blq, d};
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t blq = generic_op->localOpConfig.getRoot()["blq"];
        int stage = generic_op->localOpConfig.getRoot()["stage"];

        string expression_template;
        if (stage == 0)
        {
            // vr_, mask-> vrm
            expression_template =
                R"(@output0@[B, NQ, NV, BLK, D] = @input0@[B, NQ, NV, BLK, D] * @input1@[NQ, NV, @blq@, BLK];)";
        }
        else if (stage == 1)
        {
            //kr_, vrm -> kv
            expression_template =
                R"(@output0@[B, NQ, NV, KD, D] +=! @input0@[B, NQ, BLK, KD] * @input1@[B, NQ, NV, BLK, D];)";
        }
        else if (stage == 2)
        {
            // kv_state, cross_decay, kv -> new_kv_state
            expression_template =
                R"(@output0@[B, NQ, NV, KD, D] = @input0@[B, NQ, NV, KD, D] * @input1@[NQ, NV] + @input2@[B, NQ, NV, KD, D];)";
        }
        else if (stage == 3)
        {
            // qr, new_kv_reccurent, inner_decay, attn->out
            // expression_template =
            //     R"(mediate0[B, NBL, NQ, NV, BLQ, D] +=! @input0@[B, NBL, NQ, BLQ, KD] * @input1@[B, NBL, NQ, NV, KD, D];
            //         mediate1[B, NBL, NQ, NV, BLQ, D] = mediate0[B, NBL, NQ, NV, BLQ, D] * @input2@[NQ, NV, BLQ];
            //         @output0@[B, NBL, NQ, NV, BLQ, D] = mediate1[B, NBL, NQ, NV, BLQ, D] + @input3@[B, NBL, NQ, NV, BLQ, D];
            //         )";
            expression_template =
                R"(mediate0[B, NQ, NV, BLQ, D] +=! @input0@[B, NQ, BLQ, KD] * @input1@[B, NQ, NV, KD, D]; @output0@[B, NQ, NV, BLQ, D] = mediate0[B, NQ, NV, BLQ, D] * @input2@[NQ, NV, BLQ];)";
        }
        else
        {
            NNFUSION_CHECK_FAIL() << "Incorrect Stage ID: " << stage;
        }
        std::string expression_code = op::create_code_from_template(expression_template,
                                                                    {
                                                                        {"blq", blq},
                                                                    });

        if ((stage == 1 || stage == 3) &&
            curr->get_output_element_type(0) == nnfusion::element::f16)
        {
            expression_code += "## @: tensorCoreConfig=(3, 4)";
        }
        return expression_code;
    });
