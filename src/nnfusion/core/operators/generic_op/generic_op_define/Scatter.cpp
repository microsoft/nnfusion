// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

#define REGISTER_SCATTER_OP(name)                                                                  \
    REGISTER_OP(name).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {                 \
        NNFUSION_CHECK(gnode->get_input_size() == 3);                                              \
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr()); \
        auto ref_type = gnode->get_input_element_type(0);                                          \
        auto ref_shape = gnode->get_input_shape(0);                                                \
        auto indicies_type = gnode->get_input_element_type(1);                                     \
        auto indicies_shape = gnode->get_input_shape(1);                                           \
        auto update_type = gnode->get_input_element_type(2);                                       \
        auto update_shape = gnode->get_input_shape(2);                                             \
        NNFUSION_CHECK(ref_type == update_type)                                                    \
            << "Variable should have same datetype as Update.";                                    \
        NNFUSION_CHECK(indicies_shape.size() + ref_shape.size() - 1 == update_shape.size())        \
            << "Requires updates.shape = indices.shape + ref.shape[1:].";                          \
        for (size_t i = 0; i < indicies_shape.size(); i++)                                         \
            NNFUSION_CHECK(indicies_shape[i] == update_shape[i])                                   \
                << "Requires updates.shape = indices.shape + ref.shape[1:].";                      \
        for (size_t i = 1; i < ref_shape.size(); i++)                                              \
            NNFUSION_CHECK(ref_shape[i] == update_shape[i - 1 + indicies_shape.size()])            \
                << "Requires updates.shape = indices.shape + ref.shape[1:].";                      \
        Shape null_out;                                                                            \
        null_out.push_back(1);                                                                     \
        /*\todo(wenxh): Ensure output0 is reference to input0*/                                    \
        gnode->set_output_type_and_shape(0, ref_type, ref_shape);                                  \
    });

REGISTER_SCATTER_OP(ScatterSub)
REGISTER_SCATTER_OP(ScatterAdd)
REGISTER_SCATTER_OP(ScatterMax)
REGISTER_SCATTER_OP(ScatterMin)