// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/antares_ke_imp.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(fantares_mode);

namespace nnfusion
{
    namespace kernels
    {
        namespace hlsl
        {
            class HLSLKernelEmitter : public KernelEmitter
            {
            public:
                HLSLKernelEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "hlsl")
                {
                }
                virtual LanguageUnit_p emit_dependency() override;
            };

            class AntaresHLSLKernelEmitter : public HLSLKernelEmitter
            {
            public:
                AntaresHLSLKernelEmitter(shared_ptr<KernelContext> ctx)
                    : HLSLKernelEmitter(ctx)
                    , m_antares_ke_imp(new AntaresKEImp)
                {
                    if (FLAGS_fantares_mode)
                    {
                        auto ir = nnfusion::op::get_translation(ctx->gnode);
                        if (!ir.empty())
                        {
                            auto info = m_antares_ke_imp->autogen(ir);
                            antares_code = info.first;
                            m_is_tuned = info.second;

                            std::string annotation = nnfusion::op::get_annotation(ir);
                            // if is_memcpy, no need to request antares server
                            if (annotation.find("|memcpy|") != string::npos)
                            {
                                is_memcpy = true;
                            }
                        }
                        if (ir.empty())
                        {
                            static std::unordered_set<std::string> log_cache;
                            if (log_cache.count(ctx->gnode->get_op_type()) == 0)
                            {
                                NNFUSION_LOG(INFO) << "No Antares Translation for Op: "
                                                   << ctx->gnode->get_op_type();
                                log_cache.insert(ctx->gnode->get_op_type());
                            }

                            auto& op_reg =
                                nnfusion::op::lookup_op_config(ctx->gnode->get_op_type());
                            if (op_reg.f_kernel_funcs.count("HLSL") != 0)
                            {
                                antares_code = op_reg.f_kernel_funcs["HLSL"](ctx->gnode);
                            }
                        }

                        kernel_info =
                            nnfusion::kernels::AntaresKEImp::get_kernel_info(antares_code);
                        NNFUSION_CHECK(!kernel_info.empty())
                            << "Can not extract kernel info from antares response: \n antares IR: "
                            << ir << "\n antares response: " << antares_code;
                        process_antares_kernel_info();
                    }
                }

                bool is_eliminative() override;
                virtual LanguageUnit_p emit_function_body() override;
                virtual LanguageUnit_p emit_function_call() override;

                AntaresKEImp::Pointer m_antares_ke_imp;
                std::string antares_code;
                bool is_memcpy = false;

            protected:
                // map tensor names and allocate tmp tensor
                void process_antares_kernel_info();
                void find_launch_config(const std::string& str,
                                        std::map<std::string, std::string>& symbol_expr,
                                        std::string& block_num);
                std::string ir, options;
                std::vector<AntaresKernelInfo::Pointer> kernel_info;
                std::unordered_map<std::string, std::string>
                    tensor_name_map; // antares tensor name : kernel tensor name
            };
        } // namespace hlsl
    }     // namespace kernels
} // namespace nnfusion
