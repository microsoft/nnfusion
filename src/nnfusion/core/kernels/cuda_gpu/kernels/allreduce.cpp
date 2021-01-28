// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class SuperScalerAllReduce : public KernelEmitter
            {
            public:
                string tensor_name;
                SuperScalerAllReduce(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "SuperScaler")
                {
                    tensor_name = ctx->gnode->get_name();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto input0_size = m_context->inputs.front()->size(false);
                    auto input0_allocated_bytes = m_context->inputs.front()->size(true);
                    auto code = nnfusion::op::create_code_from_template(
                        R"(sc_allreduce("@tensorname@", input0, @input0_size@, stream);
if(input0==output0) return;
CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0, @input0_allocated_bytes@, cudaMemcpyDefault, stream));
)",
                        {{"input0_size", input0_size},
                         {"input0_allocated_bytes", input0_allocated_bytes},
                         {"tensorname", tensor_name}});
                    // allreduce and applygradient use the same stream.
                    lu << code;
                    return _lu;
                }

                LanguageUnit_p emit_function_signature()
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
                    auto& lu = *_lu;

                    vector<string> params;
                    for (size_t i = 0; i < m_context->inputs.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
                        ss << "input" << i;
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->outputs.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
                        ss << "output" << i;
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->tensors.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
                        ss << m_context->tensors[i]->get_name();
                        params.push_back(ss.str());
                    }

                    lu << "void "
                       << "(cudaStream_t stream, " << join(params, ", ") << ")";
                    return _lu;
                }

                LanguageUnit_p emit_dependency()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::superscaler);
                    return _lu;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("AllReduce",                                               //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::SuperScalerAllReduce)                                // constructor
