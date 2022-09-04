// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cuda_codegen_pass.hpp"

namespace nnfusion
{
    namespace codegen
    {
        class CudaMultiCodegenPassPre : public CudaCodegenPass
        {
        public:
            CudaMultiCodegenPassPre(
                const std::string& codegen_folder = "./nnfusion_rt/cuda_codegen/",
                const std::string& kernel_folder = "./nnfusion_rt/cuda_codegen/",
                const std::string kernel_suffix = ".cu")
                : CudaCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

            virtual bool run(std::shared_ptr<InterpreterContext> ctx,
                             std::shared_ptr<TranslationUnit> tu) override;

            CodeGenerator::Pointer get_projgen() { return this->projgen; }
        };

        class CudaMultiCodegenPass : public BaseCodegenPass
        {
        public:
            CudaMultiCodegenPass(const std::string& codegen_folder = "./nnfusion_rt/cuda_codegen/",
                                 const std::string& kernel_folder = "./nnfusion_rt/cuda_codegen/",
                                 const std::string kernel_suffix = ".cu")
                : BaseCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }
        };
    } // namespace codegen
} // namespace nnfusion
