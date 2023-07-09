// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This a new interface for Engine

#pragma once
#include "nnfusion/engine/engine.hpp"

namespace nnfusion
{
    namespace engine
    {
        class HLSLEngine : public Engine
        {
        public:
            HLSLEngine();
        };
    } // namespace engine
} // namespace nnfusion