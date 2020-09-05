// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "base_codegen_pass.hpp"
#include "codegenerator.hpp"
#include "nnfusion/engine/interpreter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::codegen;

namespace nnfusion
{
    namespace codegen
    {
        class HLSLCodegenPass : public BaseCodegenPass
        {
        public:
            HLSLCodegenPass(
                const std::string& codegen_folder = "./nnfusion_rt/dxcompute_codegen/",
                const std::string& kernel_folder = "./nnfusion_rt/dxcompute_codegen/HLSL/",
                const std::string kernel_suffix = ".hlsl")
                : BaseCodegenPass(codegen_folder, kernel_folder, kernel_suffix)
            {
            }

        protected:
            virtual void initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu) override;
            virtual bool collect_mem(std::shared_ptr<InterpreterContext> ctx,
                                     std::shared_ptr<TranslationUnit> tu) override
            {
                return true;
            }
            virtual NNFusion_DeviceType device_type() override { return NNFusion_DeviceType::HLSL; }
            inline void UNHANDLED_CASE(std::shared_ptr<GNode> curr)
            {
                printf("## Unhandled case for %s:\n", curr->get_op_ptr()->get_op_type().c_str());
                for (int i = 0; i < curr->get_input_size(); ++i)
                    printf(
                        ">> in-%d : %s\n", i, vector_to_string(curr->get_input_shape(i)).c_str());
                for (int i = 0; i < curr->get_output_size(); ++i)
                    printf(
                        ">> out-%d: %s\n", i, vector_to_string(curr->get_output_shape(i)).c_str());
                exit(1);
            };
        };
    }
}