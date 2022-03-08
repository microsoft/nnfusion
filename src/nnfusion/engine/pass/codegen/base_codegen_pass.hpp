// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "codegenerator.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/pass/tensor/tensor_memory_layout.hpp"
using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::codegen;

namespace nnfusion
{
    namespace codegen
    {
        class BaseCodegenPass : public IInterpreterPass
        {
        public:
            BaseCodegenPass(const std::string& codegen_folder = "./nnfusion_rt/base_codegen/",
                            const std::string& kernel_folder = "./nnfusion_rt/base_codegen/",
                            const std::string kernel_suffix = ".cpp")
                : projgen(new CodeGenerator(codegen_folder, kernel_suffix))
                , m_codegen_folder(codegen_folder)
                , m_kernel_folder(kernel_folder)
                , m_kernel_suffix(kernel_suffix)
            {
            }
            virtual bool run(std::shared_ptr<InterpreterContext> ctx,
                             std::shared_ptr<TranslationUnit> tu) override;

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu);
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu);
            virtual bool collect_mem(std::shared_ptr<InterpreterContext> ctx,
                                     std::shared_ptr<TranslationUnit> tu);
            virtual bool collect_stream(std::shared_ptr<InterpreterContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu)
            {
                return true;
            }
            virtual bool modify_codegen() { return true; }
            virtual bool after_projgen();
            virtual nnfusion::codegen::CodegenFuncCallsUnit_p
                get_kernel_func_calls(const string& calls_symbol,
                                      CodegenMainBlockUnit_p main_block);
            virtual void change_codegen_folder(const std::string& codegen_folder)
            {
                m_codegen_folder = codegen_folder;
                projgen->change_codegen_folder(codegen_folder);
            }
            const std::string& get_codegen_folder() const { return m_codegen_folder; }
            virtual void change_kernel_folder(const std::string& kernel_folder)
            {
                m_kernel_folder = kernel_folder;
            }
            virtual void change_kernel_suffix(const std::string& kernel_suffix)
            {
                m_kernel_suffix = kernel_suffix;
                projgen->change_kernel_suffix(kernel_suffix);
            }
            const std::string& get_kernel_suffix() const { return m_kernel_suffix; }
            const std::string& get_kernel_folder() const { return m_kernel_folder; }
            void separate_func_defs_files(int file_number, const std::string& kernel_folder);
            void add_init_and_exit_pair(LanguageUnit_p lup_in_init, LanguageUnit_p lup_in_exit);

            template <typename LanguageUnitType1, typename LanguageUnitType2>
            std::pair<std::shared_ptr<LanguageUnitType1>, std::shared_ptr<LanguageUnitType2>>
                create_init_and_exit_pair(const string& init_symbol,
                                          const string& exit_symbol,
                                          const string& init_pwd = "",
                                          const string& exit_pwd = "",
                                          const string& init_write_to = "",
                                          const string& exit_write_to = "")
            {
                std::shared_ptr<LanguageUnitType1> lup_in_init =
                    std::make_shared<LanguageUnitType1>(init_symbol);
                lup_in_init->pwd = init_pwd;
                lup_in_init->write_to = init_write_to;

                std::shared_ptr<LanguageUnitType2> lup_in_exit =
                    std::make_shared<LanguageUnitType2>(exit_symbol);
                lup_in_exit->pwd = exit_pwd;
                lup_in_exit->write_to = exit_write_to;

                add_init_and_exit_pair(lup_in_init, lup_in_exit);

                return std::make_pair(lup_in_init, lup_in_exit);
            }
            virtual NNFusion_DeviceType device_type() { return NNFusion_DeviceType::UNKNOWN; }
            virtual std::pair<LanguageUnit_p, LanguageUnit_p>
                get_customized_mem_imp(nnfusion::ir::Instruction::Pointer ins);
            LanguageUnit_p codegen_mem_ref(KernelEmitter::Pointer kernel);
            LanguageUnit_p codegen_device_type();
            LanguageUnit_p codegen_workspace_size(std::shared_ptr<TranslationUnit> tu);
            CodeGenerator::Pointer projgen;
            std::unordered_map<string, nnfusion::codegen::CodegenFuncCallsUnit_p> kernel_func_calls;
            std::unordered_map<string, std::pair<std::string, LanguageUnit_p>> kernel_func_defs;
            std::string m_codegen_folder;
            std::string m_kernel_folder;
            std::string m_kernel_suffix;
            std::unordered_set<std::shared_ptr<nnfusion::descriptor::Tensor>> free_at_last;
        };
    }
}