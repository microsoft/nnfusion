// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/antares_ke_imp.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fantares_codegen_server);

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class CpuKernelEmitter : public KernelEmitter
            {
            public:
                CpuKernelEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cpu")
                {
                }
                LanguageUnit_p emit_function_signature() override;
            };

            class MklKernelEmitter : public CpuKernelEmitter
            {
            public:
                MklKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                }
            };

            class EigenKernelEmitter : public CpuKernelEmitter
            {
            public:
                EigenKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                }

                LanguageUnit_p emit_eigen_utils();

            protected:
                std::string emit_eigen_vector(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                              const string& name = "");
                std::string emit_eigen_matrix(const shared_ptr<nnfusion::descriptor::Tensor> tw,
                                              const string& name = "");
            };

            class MlasKernelEmitter : public CpuKernelEmitter
            {
            public:
                MlasKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                }
            };

            class AntaresCpuKernelEmitter : public CpuKernelEmitter
            {
            public:
                AntaresCpuKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                    , m_antares_ke_imp(new AntaresKEImp)
                {
                    m_intra_op_parallelism = true;
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
                    }
                }

                bool is_eliminative() override;
                virtual LanguageUnit_p emit_function_body() override;
                virtual LanguageUnit_p emit_dependency() override;

                AntaresKEImp::Pointer m_antares_ke_imp;
                std::string antares_code;
                bool is_memcpy = false;
            };

            class CustomCPUKernelEmitter : public CpuKernelEmitter
            {
            public:
                CustomCPUKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    auto op_config =
                        nnfusion::op::lookup_op_config(m_context->gnode->get_op_type());
                    is_memcpy = op_config.get("is_memcpy");
                }

                virtual shared_ptr<nnfusion::cache::KernelEntry> get_kernel_cache_entry(
                    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry = nullptr) override
                {
                    if (kernel_entry == nullptr)
                    {
                        kernel_entry = std::make_shared<nnfusion::cache::KernelEntry>();
                    }
                    if (kernel_entry->source == "")
                    {
                        kernel_entry->source = "Custom";
                    }
                    return CpuKernelEmitter::get_kernel_cache_entry(kernel_entry);
                }

                bool is_eliminative() override
                {
                    return (is_memcpy &&
                            m_context->inputs[0]->is_same_address(m_context->outputs[0]));
                }
                LanguageUnit_p emit_function_body() override
                {
                    std::string body_code;
                    auto op_config =
                        nnfusion::op::lookup_op_config(m_context->gnode->get_op_type());
                    if (op_config.f_kernel_funcs.count("GENERIC_CPU") > 0)
                        body_code = op_config.f_kernel_funcs["GENERIC_CPU"](m_context->gnode);
                    if (body_code == "")
                        return nullptr;
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << body_code << "\n";
                    lu.block_end();
                    return _lu;
                }
                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

                bool is_memcpy = false;
            };

            class SimdKernelEmitter : public CpuKernelEmitter
            {
            public:
                SimdKernelEmitter(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                {
                    m_intra_op_parallelism = true;
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel()
                {
                    return std::make_pair("", nullptr);
                }

            protected:
                const uint32_t m_simd_block_size = 8;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
