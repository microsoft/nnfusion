// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

template <typename T>
std::string TranslateToIR(std::shared_ptr<graph::GNode> gnode)
{
    auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
    T start = generic_op->localOpConfig.getRoot()["start"];
    T limit = generic_op->localOpConfig.getRoot()["limit"];
    T delta = generic_op->localOpConfig.getRoot()["delta"];

    size_t num = std::max((int64_t)(std::ceil((double)(limit - start) / delta)), 0l);

    auto ir_template = R"( @output0@[N] = @start@ + N * @delta@ where N in @num@; )";

    op::OpConfig::any op_config;
    op_config["start"] = std::to_string(start);
    op_config["limit"] = std::to_string(limit);
    op_config["num"] = std::to_string(num);

    return op::create_code_from_template(ir_template, op_config);
}

REGISTER_OP(Range)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        double start = generic_op->localOpConfig.getRoot()["start"];
        double limit = generic_op->localOpConfig.getRoot()["limit"];
        double delta = generic_op->localOpConfig.getRoot()["delta"];
        size_t num = std::max((int64_t)(std::ceil((double)(limit - start) / delta)), 0l);
        nnfusion::Shape output_shape_0;
        output_shape_0.push_back(num);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto element_type = gnode->get_input_element_type(0);
        if (element_type == element::f32 || element_type == element::f64)
            return TranslateToIR<double>(gnode);
        else if (element_type == element::i32 || element_type == element::i64)
            return TranslateToIR<int64_t>(gnode);
        else
            NNFUSION_CHECK_FAIL() << "non-supported data type for Range op: "
                                  << element_type.c_type_string();
    });