// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "product.hpp"

using namespace nnfusion::op;

Product::Product(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("Product", reduction_axes)
{
}
