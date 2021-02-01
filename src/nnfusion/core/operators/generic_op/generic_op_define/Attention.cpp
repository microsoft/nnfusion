// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Attention)
    .attr<int>("num_heads")
    .attr<int>("batch_size")
    .attr<int>("sequence_length")
    .attr<int>("past_sequence_length", 0)
    .attr<int>("head_size")
    .attr<bool>("unidirectional", false) // Whether every token can only attend to previous tokens
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        /*
        Input0, "input", "3D input tensor with shape (batch_size * sequence_length, hidden_size), hidden_size = num_heads * head_size", "float/float16"
        Input1, "mask_index", "Attention mask with shape (batch_size, past_sequence_length + sequence_length), or index with shape (batch_size) or (2 * batch_size).", "int32", Optional
        Input2, "past", "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).", "float/float16", Optional
        Output0, "output", "3D output tensor with shape (batch_size, append_length, hidden_size)", "float/float16"
        Output1, "present", "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)", "float/float16", Optional
        */
        auto input_size = gnode->get_input_size();
        NNFUSION_CHECK(input_size >= 1) << "QkvtoCtx should have at least 1 inputs.";

        auto input_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(input_shape.size() == 2) << "input shall be 2 dimensions, got "
                                                << input_shape.size();

        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto& cfg = generic_op->localOpConfig.getRoot();
        int num_heads = cfg["num_heads"];
        int batch_size = cfg["batch_size"];
        int sequence_length = cfg["sequence_length"];
        int past_sequence_length = cfg["past_sequence_length"];
        int head_size = cfg["head_size"];
        bool unidirectional = cfg["unidirectional"];

        int hidden_size = num_heads * head_size;
        NNFUSION_CHECK(hidden_size % num_heads == 0)
            << "hidden_size should be divisiable by value of the num_heads attribute.";

        nnfusion::Shape output0_shape(
            {(size_t)batch_size, (size_t)sequence_length, (size_t)hidden_size});
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output0_shape);

        if (input_size >= 2)
        {
            auto mask_shape = gnode->get_input_shape(1);
            NNFUSION_CHECK(mask_shape.size() == 1 || mask_shape.size() == 2)
                << "mask shall be 1 or 2 dimensions, got " << mask_shape.size();
            if (mask_shape.size() == 1)
            {
                NNFUSION_CHECK(mask_shape[0] == batch_size || mask_shape[0] == 2 * batch_size)
                    << "mask_index dimension 0 shall have length of batch_size or 2 * batch_size";
            }
            else
            {
                NNFUSION_CHECK(mask_shape[0] == batch_size)
                    << "mask_index dimension 0 shall have length of batch_size";
            }

            if (input_size == 3)
            {
                NNFUSION_CHECK(unidirectional == true) << "past is only allowed for unidirectional";
                auto past_shape = gnode->get_input_shape(4);
                NNFUSION_CHECK(past_shape.size() == 5) << "past shall be 5 dimensions, got "
                                                       << past_shape.size();
                NNFUSION_CHECK(past_shape[0] == 2) << "past dimension 0 shall have length of 2";
                NNFUSION_CHECK(past_shape[1] == batch_size)
                    << "past dimension 1 shall have same length as dimension 0 of input";
                NNFUSION_CHECK(past_shape[2] == num_heads)
                    << "past dimension 2 shall have length of num_heads " << num_heads;
                NNFUSION_CHECK(past_shape[4] == hidden_size / num_heads)
                    << "past dimension 2 shall have length of " << hidden_size / num_heads;

                NNFUSION_CHECK(mask_shape[1] = past_shape[3] + input_shape[1])
                    << "mask_index dimension 1 shall have length of (past_sequence_length + "
                       "sequence_length)";
                nnfusion::Shape output1_shape(
                    {2,
                     (size_t)batch_size,
                     (size_t)num_heads,
                     (size_t)past_sequence_length + (size_t)sequence_length,
                     (size_t)head_size});
                gnode->set_output_type_and_shape(
                    1, gnode->get_input_element_type(0), output1_shape);
            }
        }

    });

// REGISTER_OP(Attention)
//     .attr<int>("num_heads")              // Number of attention heads
//     .attr<bool>("unidirectional", false) // Whether every token can only attend to previous tokens
//     .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
//         /*
//         Input0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size), hidden_size = num_heads * head_size", "float/float16"
//         Input1, "weight", "2D input tensor with shape (hidden_size, 3 * hidden_size)", "float/float16"
//         Input2, "bias", "1D input tensor with shape (3 * hidden_size)", "float/float16"
//         Input3, "mask_index", "Attention mask with shape (batch_size, past_sequence_length + sequence_length), or index with shape (batch_size) or (2 * batch_size).", "int32", Optional
//         Input4, "past", "past state for key and value with shape (2, batch_size, num_heads, past_sequence_length, head_size).", "float/float16", Optional
//         Output0, "output", "3D output tensor with shape (batch_size, append_length, hidden_size)", "float/float16"
//         Output1, "present", "present state for key and value with shape (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)", "float/float16", Optional
//         */
//         auto input_size = gnode->get_input_size();
//         NNFUSION_CHECK(input_size >= 3) << "Attention should have at least 3 inputs.";

//         auto input_shape = gnode->get_input_shape(0);
//         NNFUSION_CHECK(input_shape.size() == 3) << "input shall be 3 dimensions, got "
//                                                 << input_shape.size();

//         size_t batch_size = input_shape[0];
//         size_t sequence_length = input_shape[1];
//         size_t hidden_size = input_shape[2];

//         auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
//         auto& cfg = generic_op->localOpConfig.getRoot();
//         int num_heads = cfg["num_heads"];
//         bool unidirectional = cfg["unidirectional"];

//         NNFUSION_CHECK(hidden_size % num_heads == 0)
//             << "Input 0 dimension 2 should be divisiable by value of the num_heads attribute.";

//         auto weight_shape = gnode->get_input_shape(1);
//         NNFUSION_CHECK(weight_shape.size() == 2) << "weight shall be 2 dimensions, got "
//                                                  << weight_shape.size();
//         NNFUSION_CHECK(weight_shape[0] == hidden_size)
//             << "Weight dimension 0 should have same length as dimension 2 of input";
//         NNFUSION_CHECK(weight_shape[1] == 3 * weight_shape[0])
//             << "'weights' dimension 1 should be 3 times of weight dimension 0";

//         auto bias_shape = gnode->get_input_shape(2);
//         NNFUSION_CHECK(bias_shape.size() == 1) << "bias shall be 1 dimensions, got "
//                                                << bias_shape.size();
//         NNFUSION_CHECK(bias_shape[0] == weight_shape[1])
//             << "bias dimension 0 should have same length as dimension 1 of weights";

//         gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape);
//         size_t all_sequence_length = 0;

//         if (input_size >= 4)
//         {
//             auto mask_shape = gnode->get_input_shape(3);
//             NNFUSION_CHECK(mask_shape.size() == 1 || mask_shape.size() == 2)
//                 << "mask shall be 1 or 2 dimensions, got " << mask_shape.size();
//             if (mask_shape.size() == 1)
//             {
//                 NNFUSION_CHECK(mask_shape[0] == batch_size || mask_shape[0] == 2 * batch_size)
//                     << "mask_index dimension 0 shall have length of batch_size or 2 * batch_size";
//             }
//             else
//             {
//                 NNFUSION_CHECK(mask_shape[0] == batch_size)
//                     << "mask_index dimension 0 shall have length of batch_size";
//             }

//             if (input_size == 5)
//             {
//                 NNFUSION_CHECK(unidirectional == true) << "past is only allowed for unidirectional";
//                 auto past_shape = gnode->get_input_shape(4);
//                 NNFUSION_CHECK(past_shape.size() == 5) << "past shall be 5 dimensions, got "
//                                                        << past_shape.size();
//                 NNFUSION_CHECK(past_shape[0] == 2) << "past dimension 0 shall have length of 2";
//                 NNFUSION_CHECK(past_shape[1] == batch_size)
//                     << "past dimension 1 shall have same length as dimension 0 of input";
//                 NNFUSION_CHECK(past_shape[2] == num_heads)
//                     << "past dimension 2 shall have length of num_heads " << num_heads;
//                 NNFUSION_CHECK(past_shape[4] == hidden_size / num_heads)
//                     << "past dimension 2 shall have length of " << hidden_size / num_heads;

//                 all_sequence_length = past_shape[3] + input_shape[1];
//                 nnfusion::Shape output1_shape;
//                 for (auto dim : past_shape)
//                 {
//                     output1_shape.push_back(dim);
//                 }
//                 output1_shape[3] = all_sequence_length;

//                 NNFUSION_CHECK(mask_shape[1] = all_sequence_length)
//                     << "mask_index dimension 1 shall have length of (past_sequence_length + "
//                        "sequence_length)";

//                 gnode->set_output_type_and_shape(
//                     1, gnode->get_input_element_type(0), output1_shape);
//             }
//         }

//     });
