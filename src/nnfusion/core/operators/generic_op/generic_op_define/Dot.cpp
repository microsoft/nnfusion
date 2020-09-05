// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Dot)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto shape_0 = curr->get_input_shape(0);
        auto shape_1 = curr->get_input_shape(1);
        int N = _op->get_transpose_A() ? shape_0[1] : shape_0[0];
        int K = _op->get_transpose_A() ? shape_0[0] : shape_0[1];
        int M = _op->get_transpose_B() ? shape_1[0] : shape_1[1];

        return op::create_code_from_template(
            R"( - input("input0", @shape_0@); input("input1", @shape_1@); k = loop(@K@); output(@output_shape@, lambda i, j: tvm.sum(args("input0")[@dimA@] * args("input1")[@dimB@], axis=k));)",
            {{"shape_0", vector_to_string(shape_0)},
             {"shape_1", vector_to_string(shape_1)},
             {"dimA", _op->get_transpose_A() ? "k,i" : "i,k"},
             {"dimB", _op->get_transpose_B() ? "j,k" : "k,j"},
             {"K", K},
             {"output_shape", vector_to_string(curr->get_output_shape(0))}});
    });
