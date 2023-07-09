// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

DECLARE_bool(fhost_entry);
DECLARE_bool(fcodegen_pybind);
namespace nnfusion
{
    namespace codegen
    {
        class LanguageUnitwithVec : public LanguageUnit
        {
        public:
            LanguageUnitwithVec()
                : LanguageUnit()
            {
            }

            LanguageUnitwithVec(const string symbol)
                : LanguageUnit(symbol)
            {
            }

            virtual void execute(bool append = true) override;
            virtual void collect_requirement();

            std::deque<LanguageUnit_p> unit_vec;
        };

        using LanguageUnitwithVec_p = std::shared_ptr<LanguageUnitwithVec>;

        class CodegenFuncCallsUnit : public LanguageUnitwithVec
        {
        public:
            CodegenFuncCallsUnit()
                : LanguageUnitwithVec()
            {
            }

            CodegenFuncCallsUnit(const string symbol)
                : LanguageUnitwithVec(symbol)
            {
            }
            // Wrap the func calls. Replace the CodegenFuncCallsUnit with new_caller.
            virtual LanguageUnit_p
                wrap(LanguageUnit_p new_caller, LanguageUnit_p begin, LanguageUnit_p end);
        };
        using CodegenFuncCallsUnit_p = std::shared_ptr<CodegenFuncCallsUnit>;

        class CodegenMainBlockUnit : public LanguageUnitwithVec
        {
        public:
            CodegenMainBlockUnit()
                : LanguageUnitwithVec()
            {
                begin = std::make_shared<LanguageUnit>("main_block_begin");
                end = std::make_shared<LanguageUnit>("main_block_begin");
            }

            CodegenMainBlockUnit(const string symbol)
                : LanguageUnitwithVec(symbol)
            {
                begin = std::make_shared<LanguageUnit>(symbol + "_begin");
                end = std::make_shared<LanguageUnit>(symbol + "_begin");
            }

            virtual void execute(bool append = true) override;
            virtual void collect_requirement() override;

            LanguageUnit_p begin, end;
        };
        using CodegenMainBlockUnit_p = std::shared_ptr<CodegenMainBlockUnit>;

        class CodeGenerator
        {
        public:
            CodeGenerator(const std::string& codegen_folder, const std::string kernel_suffix)
                : m_codegen_folder(codegen_folder)
                , m_kernel_suffix(kernel_suffix)
            {
                lup_codegen = std::make_shared<LanguageUnit>("codegen");
                lup_init = std::make_shared<CodegenMainBlockUnit>("codegen_init");
                lup_exec = std::make_shared<CodegenMainBlockUnit>("codegen_exec");
                lup_exit = std::make_shared<CodegenMainBlockUnit>("codegen_exit");
                lup_codegen->require(lup_init);
                lup_codegen->require(lup_exec);
                lup_codegen->require(lup_exit);
                lup_exit->require(lup_exec);
                lup_exec->require(lup_init);

                if (FLAGS_fcodegen_pybind)
                {
                    lup_exec_py = std::make_shared<CodegenMainBlockUnit>("codegen_exec_py");
                    lup_codegen->require(lup_exec_py);
                    lup_exit->require(lup_exec_py);
                    lup_exec_py->require(lup_init);
                    lup_exec_py->require(lup_exec);
                }
            }
            LanguageUnit_p lup_codegen;
            CodegenMainBlockUnit_p lup_init, lup_exec, lup_exit;
            CodegenMainBlockUnit_p lup_exec_py;
            bool codegen();
            void change_codegen_folder(const std::string& codegen_folder)
            {
                m_codegen_folder = codegen_folder;
            }
            void change_kernel_suffix(const std::string& kernel_suffix)
            {
                m_kernel_suffix = kernel_suffix;
            }
            const std::string& get_codegen_folder() const { return m_codegen_folder; }
            bool need_shared_file() { return !files_include_shared.empty(); }
            using Pointer = std::shared_ptr<CodeGenerator>;

        protected:
            void pass_exec_info();
            std::string m_codegen_folder;
            std::string m_kernel_suffix;
            std::unordered_set<std::string> files_include_shared;
        };
    }
}