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

        class HLSLMultiCodegenPassPre : public HLSLCPPCodegenPass
        {
        public:
            HLSLMultiCodegenPassPre(
                const std::string& codegen_folder = "./nnfusion_rt/dxcompute_codegen/",
                const std::string& kernel_folder = "./nnfusion_rt/dxcompute_codegen/HLSL/",
                const std::string kernel_suffix = ".hlsl")
                : HLSLCPPCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

            virtual bool run(std::shared_ptr<InterpreterContext> ctx,
                             std::shared_ptr<TranslationUnit> tu) override;

            CodeGenerator::Pointer get_projgen() { return this->projgen; }
            bool invoke_after_projgen() { return this->after_projgen(); }
            bool move_kernel_folder(std::string kernel_folder)
            {
                struct stat s;
                std::string Direct3DWinNN_path = m_codegen_folder + std::string("Direct3DWinNN/");
                std::string Direct3DXBoxNN_path = m_codegen_folder + std::string("Direct3DXBoxNN/");
                std::string nnf_desktop_runtime_folder = Direct3DWinNN_path + "runtime/";
                std::string nnf_desktop_example_folder =
                    Direct3DWinNN_path + "nnf_desktop_example/";
                std::string nnf_xbox_runtime_folder = Direct3DXBoxNN_path + "runtime/";
                std::string nnf_xbox_example_folder = Direct3DXBoxNN_path + "nnf_xbox_example";
                if (stat(kernel_folder.c_str(), &s) == 0)
                {
                    std::string cmd;
                    // copy to Direct3DWinNN
                    cmd = std::string("cp -rf ") + kernel_folder + " " + nnf_desktop_example_folder;
                    if (0 != system(cmd.c_str()))
                    {
                        return false;
                    }
                    // copy to Direct3DXBoxNN
                    cmd = std::string("cp -rf ") + kernel_folder + " " + nnf_xbox_example_folder;
                    if (0 != system(cmd.c_str()))
                    {
                        return false;
                    }
                    // remove files
                    cmd = std::string("rm -rf ") + kernel_folder;
                    if (0 != system(cmd.c_str()))
                    {
                        return false;
                    }
                }
                return true;
            }

            bool invoke_modify_projgen() { return this->modify_codegen(); };
        };
    } // namespace codegen
} // namespace nnfusion