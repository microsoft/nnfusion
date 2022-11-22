// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ArgMin)
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            NNFUSION_CHECK(2 == gnode->get_input_size());
            auto& input_shape = gnode->get_input_shape(0);
            auto _op = static_pointer_cast<nnfusion::op::ArgMax>(gnode->get_op_ptr());
            auto axis = _op->get_reduction_axis();
            if (axis >= input_shape.size())
                NNFUSION_CHECK_FAIL() << "ArgMax: axis out of range";
            nnfusion::Shape output_shape_0;
            for (int i = 0; i < input_shape.size(); ++i)
            {
                if (i != axis)
                    output_shape_0.push_back(input_shape[i]);
            }
            if(output_shape_0.size() == 0)
                output_shape_0.push_back(1);
            gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            auto make_layout = [](const std::set<int>& axis) -> std::string
            {
                std::string ret = "";
                for (auto ax : axis)
                    ret += ", N" + std::to_string(ax);
                return "[" + (axis.empty() ? "N" : ret.substr(2)) + "]";
            };

            auto _op = static_pointer_cast<nnfusion::op::ArgMax>(curr->get_op_ptr());
            auto input_shape = curr->get_input_shape(0);
            auto axis = _op->get_reduction_axis();
            if (axis >= input_shape.size())
                NNFUSION_CHECK_FAIL() << "ArgMax: axis out of range";

            auto output_shape = curr->get_output_shape(0);
            auto output_datatype = curr->get_output_element_type(0);
            NNFUSION_CHECK(output_datatype == nnfusion::element::i64 ||
                           output_datatype == nnfusion::element::i32 ||
                           output_datatype == nnfusion::element::i16)
                << "ArgMin: output datatype must be i64, i32 or i16";
            auto output_datatype_str = output_datatype.c_type_string();
            // remove _t in output_datatype_str, example: int64_t to int64
            output_datatype_str = output_datatype_str.substr(0, output_datatype_str.size() - 2);
            // make layout from shape
            std::set<int> reduced_axis;
            for (int i = 0; i < input_shape.size(); ++i)
            {
                if (i != axis)
                    reduced_axis.insert(i);
            }
            auto mediate0_layout = make_layout(reduced_axis);
            auto output0_layout = make_layout(reduced_axis);
            auto input0_layout =
                vector_to_string<std::vector<std::string>> (op::create_layout_from_dims(input_shape));
            auto ir_template =
                R"( mediate0@mediate0_layout@ <=! @input0@@input0_layout@;@output0@@output0_layout@ <=! @index0@.when([mediate0@mediate0_layout@ == @input0@@input0_layout@], @index0@.val()).cast(`@output_datatype@`); )";
            op::OpConfig::any op_config;
            op_config["input0_layout"] = input0_layout;
            op_config["mediate0_layout"] = mediate0_layout;
            op_config["index0"] = axis ? "N1" : "N0";
            op_config["output0_layout"] = output0_layout;
            op_config["output_datatype"] = output_datatype_str;
            auto expression = op::create_code_from_template(ir_template, op_config);
            NNFUSION_LOG(INFO) << expression;
            return expression;
        });