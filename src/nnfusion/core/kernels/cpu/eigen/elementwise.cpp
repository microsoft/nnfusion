// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "elementwise.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER(                                                                       \
        "" #OP_NAME "",                                                                            \
        Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4),                 \
        cpu::ElementwiseEigen<nnfusion::op::OP_NAME>);

REGISTER_EW_KERNEL(Abs)
REGISTER_EW_KERNEL(Acos)
REGISTER_EW_KERNEL(Asin)
REGISTER_EW_KERNEL(Atan)
REGISTER_EW_KERNEL(Ceiling)
REGISTER_EW_KERNEL(Cos)
REGISTER_EW_KERNEL(Cosh)
REGISTER_EW_KERNEL(Exp)
REGISTER_EW_KERNEL(Floor)
REGISTER_EW_KERNEL(Log)
REGISTER_EW_KERNEL(Sin)
REGISTER_EW_KERNEL(Sinh)
REGISTER_EW_KERNEL(Sqrt)
REGISTER_EW_KERNEL(Tan)
REGISTER_EW_KERNEL(Tanh)
REGISTER_EW_KERNEL(Power)
REGISTER_EW_KERNEL(Subtract)
REGISTER_EW_KERNEL(Divide)
REGISTER_EW_KERNEL(Rsqrt)
REGISTER_EW_KERNEL(Square)
REGISTER_EW_KERNEL(Add)
REGISTER_EW_KERNEL(Multiply)
REGISTER_EW_KERNEL(Sigmoid)
