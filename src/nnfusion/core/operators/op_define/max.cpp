// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "max.hpp"

using namespace nnfusion::op;

Max::Max(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Max", reduction_axes)
{
}
