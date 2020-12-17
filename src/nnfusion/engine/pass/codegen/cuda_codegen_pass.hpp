// Microsoft (c) 2020, NNFusion Team

#pragma once

#include <libgen.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "base_codegen_pass.hpp"
#include "codegenerator.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/interpreter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::codegen;

namespace nnfusion
{
    namespace codegen
    {
        class CudaCodegenPass : public BaseCodegenPass
        {
        public:
            CudaCodegenPass(const std::string& codegen_folder = "./nnfusion_rt/cuda_codegen/",
                            const std::string& kernel_folder = "./nnfusion_rt/cuda_codegen/",
                            const std::string kernel_suffix = ".cu")
                : BaseCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual void set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu);
            virtual bool collect_stream(std::shared_ptr<InterpreterContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_mem(std::shared_ptr<InterpreterContext> ctx,
                                     std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu) override;
            virtual std::vector<std::pair<string, vector<nnfusion::ir::Instruction::Pointer>>>
                collect_ins(std::shared_ptr<InterpreterContext> ctx,
                            std::shared_ptr<TranslationUnit> tu);
            virtual void create_graph_config(std::shared_ptr<InterpreterContext> ctx,
                                             std::shared_ptr<TranslationUnit> tu);
            virtual void create_header_file(std::shared_ptr<InterpreterContext> ctx,
                                            std::shared_ptr<TranslationUnit> tu);
            virtual void create_main_file(std::shared_ptr<InterpreterContext> ctx,
                                          std::shared_ptr<TranslationUnit> tu);
            virtual void create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu);
            virtual bool modify_codegen() override;
            virtual NNFusion_DeviceType device_type() override
            {
                return NNFusion_DeviceType::CUDA_GPU;
            }
            virtual std::string get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu);
            virtual std::pair<std::string, std::string>
                get_paras_and_args(std::vector<nnfusion::ir::Instruction::Pointer>& ir_vec);
            virtual nnfusion::LanguageUnit_p
                func_call_codegen(nnfusion::ir::Instruction::Pointer ins,
                                  bool func_call_only = false,
                                  const std::string& func_call = "");

            nnfusion::async::HostAsyncManager* host_async_manager;
            nnfusion::async::DeviceStreamAsyncManager* device_async_manager;
            unordered_set<string> global_required;
            bool superscaler_enable = false;
        };
    }
}
