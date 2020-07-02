// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "min.hpp"

using namespace nnfusion::op;

Min::Min(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Min", reduction_axes)
{
}
