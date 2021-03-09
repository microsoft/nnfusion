// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_common_ops.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "matmuladd.hpp"
namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class ElementWise : public CudaElementwiseEmitter
            {
            public:
                friend class nnfusion::kernels::cuda::MatMulAdd;
                ElementWise(shared_ptr<KernelContext> ctx)
                    : CudaElementwiseEmitter(ctx)
                {
                    NNFUSION_CHECK(ctx->outputs.size() == 1)
                        << "Multi-output elementwise ops are not currently supported.";

                    for (auto arg : ctx->inputs)
                    {
                        data_types.push_back(arg->get_element_type().c_type_string());
                    }
                    data_types.push_back(ctx->outputs[0]->get_element_type().c_type_string());
                }

                LanguageUnit_p emit_function_body() override
                {
                    create_ptr(LanguageUnit, lu_, get_function_name());
                    LanguageUnit& lu = *lu_;

                    //std::string op = CudaOpMap<T>::op;
                    auto iter = CudaElementOpMap.find(m_context->gnode->get_op_type());
                    NNFUSION_CHECK(iter != CudaElementOpMap.end())
                        << "unable find op type: " << m_context->gnode->get_op_type();
                    std::string op = iter->second.op;

                    if (m_context->gnode->get_op_type() == "Convert")
                    {
                        lu.require(declaration::cuda_convert_template);
                        lu.require(header::cublas);
                    }
                    else if (iter->second.math_kernel != "")
                    {
                        auto math_kernel =
                            get_math_kernel(op, iter->second.math_kernel, data_types);
                        NNFUSION_CHECK_NOT_NULLPTR(math_kernel);
                        lu.require(math_kernel);
                        if (m_context->gnode->get_op_type() == "Gelu")
                        {
                            math_kernel->require(declaration::math_Gelu);
                            math_kernel->require(header::cublas);
                        }
                    }

                    auto num_inputs = data_types.size() - 1;
                    uint32_t nthreads = static_cast<uint32_t>(
                        nnfusion::shape_size(m_context->outputs[0]->get_shape()));
                    NNFUSION_CHECK(num_inputs > 0)
                        << "At least one input and one output tesnor for elementwise-op.";

                    int grids, blocks, bound;
                    compute_best_config(grids, blocks, bound);

                    {
                        std::string tid =
                            "blockIdx.x * " + std::to_string(blocks) + " + threadIdx.x";
                        if (grids == 1)
                            tid = "threadIdx.x";
                        if (bound)
                            lu << "if (" << tid << " >= " << bound << ") return;";

                        {
                            std::string invoke_func = op;
                            if (m_context->gnode->get_op_type() == "Convert")
                            {
                                invoke_func +=
                                    "<" + data_types.at(0) + ", " + data_types.at(1) + ">";
                            }
                            lu << "output0[" << tid << "] = " << invoke_func << "(";
                            for (size_t i = 0; i < num_inputs - 1; i++)
                            {
                                lu << "input" << i << "[" << tid << "], ";
                            }
                            lu << "input" << num_inputs - 1 << "[" << tid << "]);\n";
                        }
                    }
                    return lu_;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::stdio);

                    return _lu;
                }

                void set_launch_config() override
                {
                    int grids, blocks, bound;
                    compute_best_config(grids, blocks, bound);

                    m_gridDim = dim3(grids, 1, 1);
                    m_blockDim = dim3(blocks, 1, 1);
                }

                std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel() override
                {
                    //std::string op = CudaOpMap<T>::op;
                    auto iter = CudaElementOpMap.find(m_context->gnode->get_op_type());
                    NNFUSION_CHECK(iter != CudaElementOpMap.end())
                        << "unable find op type: " << m_context->gnode->get_op_type();
                    std::string op = iter->second.op;
                    shared_ptr<LanguageUnit> kernel = nullptr;

                    if (iter->second.math_kernel != "")
                    {
                        kernel = get_math_kernel(op, iter->second.math_kernel, data_types);
                        NNFUSION_CHECK_NOT_NULLPTR(kernel);
                    }
                    return std::make_pair(op, kernel);
                }

            private:
                void compute_best_config(int& grids, int& blocks, int& bound)
                {
                    uint32_t num_ele = static_cast<uint32_t>(
                        nnfusion::shape_size(m_context->outputs[0]->get_shape()));
                    for (int i = 512; i >= 64; i >>= 1)
                    {
                        if (num_ele % i == 0)
                        {
                            grids = num_ele / i, blocks = i, bound = 0;
                            return;
                        }
                    }
                    for (int i = 512; i >= 32; i--)
                    {
                        if (num_ele % i == 0)
                        {
                            grids = num_ele / i, blocks = i, bound = 0;
                            return;
                        }
                    }
                    if (num_ele < 32)
                        grids = 1, blocks = num_ele, bound = 0;
                    else
                        grids = (num_ele + 255) / 256, blocks = 256, bound = 1;
                }

                // shared_ptr<KernelContext> kernel_ctx;
                vector<string> data_types;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
