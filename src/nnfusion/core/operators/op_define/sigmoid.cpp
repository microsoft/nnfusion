// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "sigmoid.hpp"

using namespace nnfusion::op;

Sigmoid::Sigmoid()
    : ElementwiseArithmetic("Sigmoid")
{
}

SigmoidBackprop::SigmoidBackprop()
    : ElementwiseArithmetic("SigmoidBackprop")
{
}
