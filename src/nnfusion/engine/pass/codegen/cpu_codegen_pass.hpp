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
        class CpuCodegenPass : public CudaCodegenPass
        {
        public:
            CpuCodegenPass(const std::string& codegen_folder = "./nnfusion_rt/cpu_codegen/",
                           const std::string& kernel_folder = "./nnfusion_rt/cpu_codegen/",
                           const std::string kernel_suffix = ".cpp")
                : CudaCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

        protected:
            virtual void set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_stream(std::shared_ptr<InterpreterContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu) override
            {
                return true;
            }
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual void create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu) override;
            virtual void create_main_file(std::shared_ptr<InterpreterContext> ctx,
                                          std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu) override;
            virtual bool modify_codegen() override;
            virtual void create_header_file(std::shared_ptr<InterpreterContext> ctx,
                                            std::shared_ptr<TranslationUnit> tu) override;
            virtual NNFusion_DeviceType device_type() { return NNFusion_DeviceType::GENERIC_CPU; }
            bool need_intra_node_threadpool = false;
            int numa_node_num;
            unordered_map<std::string, int> cpu_kernel_thread_idx;
        };
    }
}