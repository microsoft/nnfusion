// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER(                                                                       \
        "" #OP_NAME "",                                                                            \
        Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4),                 \
        cpu::ReduceEigen<nnfusion::op::OP_NAME>);

//REGISTER_EW_KERNEL(Sum)
//REGISTER_EW_KERNEL(Product)
//REGISTER_EW_KERNEL(Max)
//REGISTER_EW_KERNEL(Min)
