// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This a new interface for Engine

#pragma once
#include "nnfusion/engine/engine.hpp"

namespace nnfusion
{
    namespace engine
    {
        class CudaEngine : public Engine
        {
        public:
            CudaEngine();
        };

        class CudaMultiEngine : public CudaEngine
        {
        public:
            CudaMultiEngine();
            bool run_on_graphs(std::vector<graph::Graph::Pointer> graphs,
                               EngineContext::Pointer context = nullptr);

        private:
            void remove_extern_c(std::string fname);
            std::string get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu, bool is_host);
            std::string get_kernel_entry_args(std::shared_ptr<TranslationUnit> tu, bool is_host);
            bool erase_all_codegen();
        };
    } // namespace engine
} // namespace nnfusion
