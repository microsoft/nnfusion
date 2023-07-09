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
                lup_main = std::make_shared<LanguageUnit>("codegen_main");
                lup_header = std::make_shared<LanguageUnit>("codegen_header");
            }

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu) override;
            virtual NNFusion_DeviceType device_type() override { return NNFusion_DeviceType::HLSL; }
            // virtual void generate_main(std::shared_ptr<InterpreterContext> ctx,
            //                            std::shared_ptr<TranslationUnit> tu);
            virtual void create_main_file(std::shared_ptr<InterpreterContext> ctx,
                                          std::shared_ptr<TranslationUnit> tu) override;
            virtual void create_header_file(std::shared_ptr<InterpreterContext> ctx,
                                            std::shared_ptr<TranslationUnit> tu) override;
            std::string get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu,
                                               bool is_host = false) override;
            void set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu) override;
            virtual bool after_projgen() override;
            virtual LanguageUnit_p get_d2hcopy(std::shared_ptr<TranslationUnit> tu) override;
            virtual LanguageUnit_p get_h2dcopy(std::shared_ptr<TranslationUnit> tu) override;
            virtual LanguageUnit_p get_sync() override;
            LanguageUnit_p lup_main, lup_header;
        };
    }
}