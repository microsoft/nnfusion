// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pad.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Pad",                                                                     // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::Pad<float>)                                                           // constructor
