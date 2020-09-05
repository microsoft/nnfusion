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
                        ir = nnfusion::op::get_translation_v2(ctx->gnode);
                        if (ir.empty())
                            ir = nnfusion::op::get_translation(ctx->gnode);
                        if (!ir.empty())
                        {
                            const char annotation[] = "## @annotation: ";
                            int pos = ir.find(annotation);
                            if (pos >= 0)
                            {
                                pos += sizeof(annotation) - 1;
                                options = ir.substr(pos);
                            }

                            if (options.size() > 0)
                            {
                                if (options[0] != '|')
                                    options = "|" + options;
                                if (options.back() != '|')
                                    options += "|";
                            }

                            auto info = m_antares_ke_imp->autogen(ir);
                            antares_code = info.first;
                            m_is_tuned = info.second;
                        }
                    }
                }

                virtual LanguageUnit_p emit_function_body() override;
                virtual LanguageUnit_p emit_function_call() override;

                AntaresKEImp::Pointer m_antares_ke_imp;
                std::string antares_code;

            protected:
                std::string ir, options;
            };
        }
    }
}
