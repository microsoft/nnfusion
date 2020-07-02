// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "stop_gradient.hpp"

using namespace nnfusion::op;

StopGradient::StopGradient()
    : ElementwiseArithmetic("StopGradient")
{
}
