// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Softmax).infershape(nnfusion::op::infershape::copy_shape_from_inputs);
