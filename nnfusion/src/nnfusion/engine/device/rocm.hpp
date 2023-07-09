// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This a new interface for Engine

#pragma once
#include "nnfusion/engine/engine.hpp"

namespace nnfusion
{
    namespace engine
    {
        class ROCmEngine : public Engine
        {
        public:
            ROCmEngine();
        };
    } // namespace engine
} // namespace nnfusion