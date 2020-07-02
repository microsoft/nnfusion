// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sum.hpp"

using namespace std;
using namespace nnfusion::op;

Sum::Sum(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Sum", reduction_axes)
{
}
