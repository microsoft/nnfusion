// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "relu.hpp"

using namespace std;
using namespace nnfusion::op;

Relu::Relu()
    : ElementwiseArithmetic("Relu")
{
}

ReluBackprop::ReluBackprop()
    : ElementwiseArithmetic("ReluBackprop")
{
}