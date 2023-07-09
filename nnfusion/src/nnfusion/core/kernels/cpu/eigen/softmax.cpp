// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Softmax",                                                                 // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::SoftmaxEigen<float>)                                                  // constructor
