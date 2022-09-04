// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This a new interface for Engine

#pragma once
#include "cuda.hpp"

namespace nnfusion
{
    namespace engine
    {
        class CudaMultiEngine : public CudaEngine
        {
        public:
            CudaMultiEngine();
            bool run_on_graphs(std::vector<graph::Graph::Pointer> graphs,
                               EngineContext::Pointer context = nullptr);

        private:
            bool erase_all_codegen();
        };
    } // namespace engine
} // namespace nnfusion