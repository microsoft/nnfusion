// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Reverse)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Reverse>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        nnfusion::Shape input_shape = gnode->get_input_shape(0);
        nnfusion::AxisSet reverse_axes = op->get_reversed_axes();
        std::vector<int> axes(input_shape.size(), 1);
        std::vector<size_t> begin(input_shape.size(), 0);
        std::vector<size_t> end = input_shape;
        for (auto i : reverse_axes)
        {
            axes[i] = -1;
            std::swap(begin[i], end[i]);
            end[i] -= (begin[i] + 1);
            begin[i] = -1;
        }

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@input_shape@, topi=topi.strided_slice(args("input0"), begin=@begin@, end=@end@, strides=@axes@)); )",
            {{"input_shape", vector_to_string(input_shape)},
             {"begin", vector_to_string(begin)},
             {"end", vector_to_string(end)},
             {"axes", vector_to_string(axes)}});
    });
