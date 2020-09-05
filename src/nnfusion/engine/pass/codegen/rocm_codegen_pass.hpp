// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cuda_codegen_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::codegen;

namespace nnfusion
{
    namespace codegen
    {
        class RocmCodegenPass : public CudaCodegenPass
        {
        public:
            RocmCodegenPass(const std::string& codegen_folder = "./nnfusion_rt/rocm_codegen/",
                            const std::string& kernel_folder = "./nnfusion_rt/rocm_codegen/",
                            const std::string kernel_suffix = ".cu")
                : CudaCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual void create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu) override;
            virtual bool after_projgen() override;
            virtual NNFusion_DeviceType device_type() { return NNFusion_DeviceType::ROCM_GPU; }
        };
    }
}