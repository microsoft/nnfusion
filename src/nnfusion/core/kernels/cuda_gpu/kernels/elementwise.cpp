// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "elementwise.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_EW_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER(                                                                       \
        "" #OP_NAME "",                                                                            \
        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("element_wise").Priority(2),             \
        cuda::ElementWise<nnfusion::op::OP_NAME>);

REGISTER_EW_KERNEL(Abs)
REGISTER_EW_KERNEL(Acos)
REGISTER_EW_KERNEL(Asin)
REGISTER_EW_KERNEL(Atan)
REGISTER_EW_KERNEL(Ceiling)
REGISTER_EW_KERNEL(Convert)
REGISTER_EW_KERNEL(Cos)
REGISTER_EW_KERNEL(Cosh)
REGISTER_EW_KERNEL(Erf)
REGISTER_EW_KERNEL(Exp)
REGISTER_EW_KERNEL(Floor)
REGISTER_EW_KERNEL(Gelu)
REGISTER_EW_KERNEL(Log)
REGISTER_EW_KERNEL(Sin)
REGISTER_EW_KERNEL(Sinh)
REGISTER_EW_KERNEL(Sqrt)
REGISTER_EW_KERNEL(Rsqrt)
REGISTER_EW_KERNEL(Square)
REGISTER_EW_KERNEL(Tan)
REGISTER_EW_KERNEL(Tanh)
REGISTER_EW_KERNEL(Power)
REGISTER_EW_KERNEL(PowerBackwardBase)
REGISTER_EW_KERNEL(PowerBackwardExponent)
REGISTER_EW_KERNEL(Subtract)
REGISTER_EW_KERNEL(Divide)
REGISTER_EW_KERNEL(DivNoNan)
REGISTER_EW_KERNEL(Sign)
REGISTER_EW_KERNEL(Convert)
REGISTER_EW_KERNEL(Equal)
REGISTER_EW_KERNEL(NotEqual)
REGISTER_EW_KERNEL(Greater)
REGISTER_EW_KERNEL(GreaterEq)
REGISTER_EW_KERNEL(Less)
REGISTER_EW_KERNEL(LessEq)
REGISTER_EW_KERNEL(Relu)
REGISTER_EW_KERNEL(Relu6)
REGISTER_EW_KERNEL(Not)
REGISTER_EW_KERNEL(Negative)
REGISTER_EW_KERNEL(Select)
REGISTER_EW_KERNEL(ReluBackprop)
REGISTER_EW_KERNEL(Relu6Backprop)
REGISTER_EW_KERNEL(And)
REGISTER_EW_KERNEL(Or)
REGISTER_EW_KERNEL(Add)
REGISTER_EW_KERNEL(Multiply)
REGISTER_EW_KERNEL(Minimum)
REGISTER_EW_KERNEL(Maximum)
REGISTER_EW_KERNEL(Nop)
REGISTER_EW_KERNEL(Sigmoid)
REGISTER_EW_KERNEL(SigmoidBackprop)
REGISTER_EW_KERNEL(GeluGrad)
