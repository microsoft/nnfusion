// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pad.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Pad",                                                     // op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT).Tag("eigen"), // attrs
                        cpu::Pad<float>)                                           // constructor