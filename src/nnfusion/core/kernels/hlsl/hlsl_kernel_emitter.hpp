// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/antares_ke_imp.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fantares_codegen_server);

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
                    if (!FLAGS_fantares_codegen_server.empty())
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
                        }
                    }
                }

                bool is_eliminative() override;
                virtual LanguageUnit_p emit_function_body() override;
                virtual LanguageUnit_p emit_function_call() override;

                AntaresKEImp::Pointer m_antares_ke_imp;
                std::string antares_code;
                bool is_memcpy = false;

            protected:
                std::string ir, options;
            };
        }
    }
}
