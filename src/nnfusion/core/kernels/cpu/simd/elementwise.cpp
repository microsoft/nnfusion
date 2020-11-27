// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "elementwise.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER(                                                                       \
        "" #OP_NAME "",                                                                            \
        Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("simd").Priority(5),                  \
        cpu::ElementwiseSimd<nnfusion::op::OP_NAME>);

REGISTER_EW_KERNEL(Abs)
REGISTER_EW_KERNEL(Ceiling)
REGISTER_EW_KERNEL(Floor)
REGISTER_EW_KERNEL(Subtract)
REGISTER_EW_KERNEL(Divide)
REGISTER_EW_KERNEL(Equal)
REGISTER_EW_KERNEL(NotEqual)
REGISTER_EW_KERNEL(Greater)
REGISTER_EW_KERNEL(GreaterEq)
REGISTER_EW_KERNEL(Less)
REGISTER_EW_KERNEL(LessEq)
REGISTER_EW_KERNEL(Rsqrt)
REGISTER_EW_KERNEL(Square)
REGISTER_EW_KERNEL(Add)
REGISTER_EW_KERNEL(Multiply)
REGISTER_EW_KERNEL(Maximum)
REGISTER_EW_KERNEL(Minimum)
REGISTER_EW_KERNEL(Sqrt)
//REGISTER_EW_KERNEL(Sign)
REGISTER_EW_KERNEL(And)
REGISTER_EW_KERNEL(Or)
REGISTER_EW_KERNEL(Not)
REGISTER_EW_KERNEL(Negative)
REGISTER_EW_KERNEL(Relu)
