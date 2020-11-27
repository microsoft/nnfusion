// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MaxPool)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::MaxPool>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        const auto& kernel = _op->get_window_shape();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        uint64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.nn.pool(args("input0"), kernel=@kernel@, stride=@stride@, padding=@padding@, pool_type="max")); )",
            {{"input_shape", vector_to_string(curr->get_input_shape(0))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))},
             {"kernel", vector_to_string(kernel)},
             {"stride", vector_to_string(stride)},
             {"padding", vector_to_string(padding)}});
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( @output0@@output0_layout@ >=! @input0@@input0_layout@ where HO in @height@, WO in @width@, KH in @stride_h@, KW in @stride_w@; )";

        auto _op = static_pointer_cast<nnfusion::op::MaxPool>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        const bool is_nchw = _op->get_data_format() == "NCHW" ? true : false;

        auto& output_shape = curr->get_output_shape(0);
        auto& strides = _op->get_window_movement_strides();

        auto output_layout = is_nchw ? std::vector<std::string>{"N", "C", "HO", "WO"}
                                     : std::vector<std::string>{"N", "HO", "WO", "C"};
        auto input_layout = std::vector<std::string>(output_layout);
        auto h_axis = is_nchw ? 2 : 1;
        auto w_axis = is_nchw ? 3 : 2;
        input_layout[h_axis] = input_layout[h_axis] + " * " + to_string(strides[0]) + " + KH";
        input_layout[w_axis] = input_layout[w_axis] + " * " + to_string(strides[1]) + " + KW";

        op::OpConfig::any config;
        config["height"] = to_string(is_nchw ? output_shape[2] : output_shape[1]);
        config["width"] = to_string(is_nchw ? output_shape[3] : output_shape[2]);
        config["stride_h"] = to_string(strides[0]);
        config["stride_w"] = to_string(strides[1]);
        config["output0_layout"] = vector_to_string<std::vector<std::string>>(output_layout);
        config["input0_layout"] = vector_to_string<std::vector<std::string>>(input_layout);

        auto expression_code = op::create_code_from_template(expression_template, config);
        return expression_code;
    });
