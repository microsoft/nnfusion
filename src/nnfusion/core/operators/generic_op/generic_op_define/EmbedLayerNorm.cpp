// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(EmbedLayerNorm)
    .attr<float>("epsilon", 1e-6)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        /*
        Input0, "input_ids", "2D words IDs with shape (batch_size, sequence_length)", "int32")
        Input1, "segment_ids", "2D segment IDs with shape (batch_size, sequence_length)", "int32")
        Input2, "word_embedding", "2D with shape (,hidden_size)", "float/float16")
        Input3, "position_embedding", "2D with shape (, hidden_size)", "float/float16")
        Input4, "segment_embedding", "2D with shape (, hidden_size)", "float/float16")
        Input5, "gamma", "1D gamma tensor for layer normalization with shape (hidden_size)", "float/float16")
        Input6, "beta", "1D beta tensor for layer normalization  with shape (hidden_size)", "float/float16")
        Input7, "mask", "2D attention mask with shape (batch_size, sequence_length)", "int32", optinal)
        Output0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "float/float16")
        Output1, "mask_index", "1D mask_index tensor with shape (batch_size)", "int32")
        */
        NNFUSION_CHECK(gnode->get_input_size() >= 7)
            << "EmbedLayerNorm should have at least 7 inputs.";

        auto input_ids_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(input_ids_shape.size() == 2) << "input_ids shall be 2 dimensions";

        auto word_embedding_shape = gnode->get_input_shape(3);
        NNFUSION_CHECK(word_embedding_shape.size() == 2) << "word_embedding shall be 2 dimensions";

        auto batch_size = input_ids_shape[0];
        auto sequence_lenth = input_ids_shape[1];
        auto hidden_size = word_embedding_shape[1];

        nnfusion::Shape output0_shape{batch_size, sequence_lenth, hidden_size};
        nnfusion::Shape output1_shape{batch_size};

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(2), output0_shape);
        gnode->set_output_type_and_shape(1, gnode->get_input_element_type(1), output1_shape);
    });
