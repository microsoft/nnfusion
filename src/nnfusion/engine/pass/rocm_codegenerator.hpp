// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    std::shared_ptr<IInterpreterPass> make_rocm_codegenerator();
} // namespace nnfusion
