// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"

REGISTER_OP(ElementWiseFused)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir = static_pointer_cast<nnfusion::op::Fused>(curr->get_op_ptr())->get_fused_ir2();
        return ir;
    });
