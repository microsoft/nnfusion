// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

#define REGISTER_CUSTOM_OP(op_x)                                                                          \
    REGISTER_OP( op_x )