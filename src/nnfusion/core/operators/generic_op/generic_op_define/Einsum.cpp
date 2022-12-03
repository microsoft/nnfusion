// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Einsum).translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
    auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
    std::vector<std::vector<std::string>> input_layout =
        generic_op->localOpConfig.getRoot()["input_layout"];
    std::vector<std::string> output_layout = generic_op->localOpConfig.getRoot()["output_layout"];

    auto join = [](std::vector<std::string> vec,
                   std::string deli = ",",
                   std::string left = "[",
                   std::string right = "]") {
        std::string ret = left;
        for (size_t i = 0; i < vec.size(); i++)
        {
            ret += vec[i];
            if (i + 1 < vec.size())
            {
                ret += deli;
            }
        }
        ret += right;
        return ret;
    };

    std::string ir = " ";
    ir +=
        "@output0@" + join(output_layout.empty() ? std::vector<std::string>{"NO"} : output_layout);
    ir += " +=! ";
    for (size_t i = 0; i < input_layout.size(); i++)
    {
        ir += "@input" + std::to_string(i) + "@" + join(input_layout[i]);
        if (i + 1 < input_layout.size())
        {
            ir += " * ";
        }
    }
    if (output_layout.empty())
    {
        ir += " where NO in 1 ";
    }
    ir += "; ";

    return ir;
});
