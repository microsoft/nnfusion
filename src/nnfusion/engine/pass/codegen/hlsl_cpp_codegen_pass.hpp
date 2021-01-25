// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "codegenerator.hpp"
#include "cuda_codegen_pass.hpp"
#include "nnfusion/engine/interpreter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::codegen;

namespace nnfusion
{
    namespace codegen
    {
        class HLSLCPPCodegenPass : public CudaCodegenPass
        {
        public:
            HLSLCPPCodegenPass(
                const std::string& codegen_folder = "./nnfusion_rt/dxcompute_codegen/",
                const std::string& kernel_folder = "./nnfusion_rt/dxcompute_codegen/HLSL/",
                const std::string kernel_suffix = ".hlsl")
                : CudaCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
                lup_main = std::make_shared<CodegenMainBlockUnit>("main");
            }

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu) override;
            virtual NNFusion_DeviceType device_type() override { return NNFusion_DeviceType::HLSL; }
            virtual void generate_main(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu);
            std::string get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu) override;
            void set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu) override;
            virtual bool after_projgen() override;
            CodegenMainBlockUnit_p lup_main;
        };
    }
}