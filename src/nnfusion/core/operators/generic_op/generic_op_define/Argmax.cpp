// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Argmax)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto ir = "temp0[B] >=! input0[B, M];"
          "@output0@[B] >=! M.when([temp0[B] == input0[B, M]], const(-1, M.dtype())) where M in M:1024;";
        return ir;
    });
