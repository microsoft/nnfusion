// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MemEffAttn)
    .attr<int>("batch_size")
    .attr<int>("num_heads")
    .attr<int>("seq_len")
    .attr<int>("seq_len_kv")
    .attr<int>("head_size")
    .attr<int>("head_size_v")
    .attr<float>("softmax_scale", 0.1580810546875)
    .attr<float>("p_dropout", 0)
    .attr<bool>("is_causal", false) // Whether every token can only attend to previous tokens
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_in_edges().size() == 3);

        auto in_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(in_shape.size() == 4);
        nnfusion::Shape outshape{in_shape[0], in_shape[2], in_shape[1], in_shape[3]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        // adhoc: to be fixed
        std::string expression_code =
            "@output0@[N0, N1, N2, N3] = @input0@[N0, N2, N1, N3] + @input1@[N0, N2, 0, N3] + "
            "@input2@[N0, N2, 0, N3];";
        return expression_code;
    });
