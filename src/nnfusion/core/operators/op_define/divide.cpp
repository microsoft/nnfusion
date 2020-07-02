// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "divide.hpp"

using namespace nnfusion::op;

Divide::Divide()
    : ElementwiseArithmetic("Divide")
{
}

DivNoNan::DivNoNan()
    : ElementwiseArithmetic("DivNoNan")
{
}