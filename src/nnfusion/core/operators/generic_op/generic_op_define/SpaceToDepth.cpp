// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(SpaceToDepth)
    .attr<nnfusion::op::OpConfig::any>("block_size")
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto shape_0 = gnode->get_input_shape(0);
        NNFUSION_CHECK(shape_0.size() == 4);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        bool is_nhwc = generic_op->localOpConfig.getRoot()["data_format"] == "NHWC";
        int block_size = generic_op->localOpConfig.getRoot()["block_size"];

        auto h = is_nhwc ? shape_0[1] : shape_0[2];
        auto w = is_nhwc ? shape_0[2] : shape_0[3];
        auto c = is_nhwc ? shape_0[3] : shape_0[1];
        NNFUSION_CHECK(h % block_size == 0 && w % block_size == 0);
        h /= block_size;
        w /= block_size;
        c *= block_size * block_size;
        shape_0[1] = is_nhwc ? h : c;
        shape_0[2] = is_nhwc ? w : h;
        shape_0[3] = is_nhwc ? c : w;

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( output0@output0_layout@ = input0@input0_layout@ where H0 in @height@, W0 in @width@, C0 in @channel@; )";

        auto input_shape = curr->get_input_shape(0);
        auto output_shape = curr->get_output_shape(0);
        auto _op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto is_nhwc = _op->localOpConfig.getRoot()["data_format"] == "NHWC";
        size_t block_size = _op->localOpConfig.getRoot()["block_size"];

        std::string output0_layout = is_nhwc ? "[N0, H0, W0, C0]" : "[N0, C0, H0, W0]";

        std::string Hfmt = "H0 * @block_size@ + (C0 // @input_channels@) \% @block_size@";
        std::string Wfmt = "W0 * @block_size@ + (C0 // @input_channels@) // @block_size@";
        std::string Cfmt = "C0 \% @input_channels@";
        std::string input0_layout = is_nhwc ? "[N0, " + Hfmt + ", " + Wfmt + ", " + Cfmt + "]"
                                            : "[N0, " + Cfmt + ", " + Hfmt + ", " + Wfmt + "]";

        nnfusion::op::OpConfig::any config;
        config["block_size"] = block_size;
        config["input_channels"] = is_nhwc ? input_shape[3] : input_shape[1];
        input0_layout = op::create_code_from_template(input0_layout, config);
        config["input0_layout"] = input0_layout;
        config["output0_layout"] = output0_layout;
        config["channel"] = is_nhwc ? output_shape[3] : output_shape[1];
        config["height"] = is_nhwc ? output_shape[1] : output_shape[2];
        config["width"] = is_nhwc ? output_shape[2] : output_shape[3];
        return op::create_code_from_template(expression_template, config);
    });
