// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DepthToSpace)
    .attr<nnfusion::op::OpConfig::any>("T")
    .attr<nnfusion::op::OpConfig::any>("block_size")
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .attr<nnfusion::op::OpConfig::any>("mode", "DCR") // choose from DCR and CRD
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto shape_0 = gnode->get_input_shape(0);
        NNFUSION_CHECK(shape_0.size() == 4);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        // NNFUSION_CHECK(generic_op->localOpConfig.getRoot()["data_format"] == "NHWC");
        bool is_nhwc = generic_op->localOpConfig.getRoot()["data_format"] == "NHWC";
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];

        auto h = (is_nhwc ? shape_0[1] : shape_0[2]) * block_size;
        auto w = (is_nhwc ? shape_0[2] : shape_0[3]) * block_size;
        auto c = (is_nhwc ? shape_0[3] : shape_0[1]) / (block_size * block_size);

        shape_0[1] = is_nhwc ? h : c;
        shape_0[2] = is_nhwc ? w : h;
        shape_0[3] = is_nhwc ? c : w;

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( mediate0@mediate0_layout@ = @input0@@input0_layout@ @cond0@; mediate1@mediate1_layout@ = mediate0@mediate0_layout@; @output0@@output0_layout@ = mediate1@mediate1o_layout@ @cond1@;)";

        auto input_shape = curr->get_input_shape(0);
        auto _op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto is_nhwc = _op->localOpConfig.getRoot()["data_format"] == "NHWC";
        auto mode = _op->localOpConfig.getRoot()["mode"];

        size_t block_size = _op->localOpConfig.getRoot()["block_size"];
        size_t c_stride = (is_nhwc ? input_shape[3] : input_shape[1]) / (block_size * block_size);
        std::string input0_c_str;

        if (mode == "CRD")
        {
            input0_c_str = op::create_code_from_template(
                "C * @block_size@ * @block_size@ + HS * @block_size@ + WS",
                {{"c_stride", to_string(c_stride)}, {"block_size", to_string(block_size)}});
        }
        else
        {
            input0_c_str = op::create_code_from_template(
                "HS * @block_size@ * @c_stride@ + WS * @c_stride@ + C",
                {{"c_stride", to_string(c_stride)}, {"block_size", to_string(block_size)}});
        }
        // auto input0_c_str = op::create_code_from_template("C * @c_stride@ * @block_size@ + HS * @c_stride@ + WS", {{"c_stride", to_string(c_stride)}, {"block_size", to_string(block_size)}});
        auto input0_layout = op::create_code_from_template(
            is_nhwc ? "[N, H, W, @c_str@]" : "[N, @c_str@, H, W]", {{"c_str", input0_c_str}});
        std::string mediate0_layout;
        if (mode == "CRD")
        {
            mediate0_layout = is_nhwc ? "[N, H, W, C, HS, WS]" : "[N, C, HS, WS, H, W]";
        }
        else
        {
            mediate0_layout = is_nhwc ? "[N, H, W, HS, WS, C]" : "[N, HS, WS, C, H, W]";
        }
        auto cond0 = op::create_code_from_template(
            "where C in @c_stride@, HS in @block_size@, WS in @block_size@",
            {{"c_stride", to_string(c_stride)}, {"block_size", to_string(block_size)}});

        auto mediate1_layout = is_nhwc ? "[N, H, HS, W, WS, C]" : "[N, C, H, HS, W, WS]";

        auto out_height = (is_nhwc ? input_shape[1] : input_shape[2]) * block_size;
        auto out_width = (is_nhwc ? input_shape[2] : input_shape[3]) * block_size;
        auto hw_str = op::create_code_from_template(
            "H // @block_size@, H % @block_size@, W // @block_size@, W % @block_size@",
            {{"block_size", to_string(block_size)}});
        auto mediate1o_layout = op::create_code_from_template(
            is_nhwc ? "[N, @hw_str@, C]" : "[N, C, @hw_str@]", {{"hw_str", hw_str}});
        auto output0_layout = is_nhwc ? "[N, H, W, C]" : "[N, C, H, W]";
        auto cond1 = op::create_code_from_template(
            "where H in @height@, W in @width@",
            {{"height", to_string(out_height)}, {"width", to_string(out_width)}});

        return op::create_code_from_template(expression_template,
                                             {{"input0_layout", input0_layout},
                                              {"cond0", cond0},
                                              {"mediate0_layout", mediate0_layout},
                                              {"mediate1_layout", mediate1_layout},
                                              {"mediate1o_layout", mediate1o_layout},
                                              {"output0_layout", output0_layout},
                                              {"cond1", cond1}});
    })
    .infersharedmemory([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];

        const Shape& input_shape = gnode->get_input_shape(0);
        std::string data_format = generic_op->localOpConfig.getRoot()["data_format"];
        bool is_nhwc = (data_format == "NHWC");
        int channel = is_nhwc ? 3 : 1;
        auto input_channel_count = input_shape[channel];

        std::vector<size_t> shared_memory;
        for (size_t i = 0; i < gnode->get_output_shape(0).size(); i++)
        {
            if (i == channel)
                shared_memory.push_back(input_channel_count / (block_size * block_size));
            else
                shared_memory.push_back(1);
        }

        generic_op->set_shared_memory(shared_memory);
    });
