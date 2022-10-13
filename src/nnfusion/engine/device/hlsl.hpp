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

        class HLSLMultiEngine : public HLSLEngine
        {
        public:
            HLSLMultiEngine();
            bool run_on_graphs(std::vector<graph::Graph::Pointer> graphs,
                               EngineContext::Pointer context = nullptr);

        private:
            bool erase_all_codegen();
        };
    } // namespace engine
} // namespace nnfusion