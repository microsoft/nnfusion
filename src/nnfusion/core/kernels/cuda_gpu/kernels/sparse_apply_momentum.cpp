// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class SparseApplyMomentum : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads, ref_offset;
                nnfusion::element::Type dtype;
                std::string name;
                std::string math_kernel;
                bool use_nesterov;
                float lr, momentum;

            public:
                SparseApplyMomentum(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[2]->size(false);
                    dtype = nnfusion::element::Type(ctx->inputs[0]->get_element_type());
                    auto var_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());

                    auto& cfg = generic_op->localOpConfig.getRoot();
                    use_nesterov = (bool)cfg["use_nesterov"];
                    lr = cfg["lr"].is_null() ? 0.001 : (float)cfg["lr"];
                    momentum = cfg["momentum"].is_null() ? 0.001 : (float)cfg["momentum"];
                    std::vector<int64_t> indices = cfg["indices"];
                    size_t N = indices.size();
                    size_t first_dim_size = m_context->inputs[0]->get_shape()[0];

                    for (size_t i = 0; i < N; i++)
                    {
                        auto index = indices[i];
                        NNFUSION_CHECK(index < first_dim_size) << "Index " << index << "at offset "
                                                               << i
                                                               << " in indices is out of range";
                    }

                    name = std::string("update_accum");
                    math_kernel = std::string("x0 * ") + to_string(momentum) + std::string(" + x1");

                    ref_offset = 1;
                    for (size_t d = 1; d < var_shape.size(); d++)
                        ref_offset *= var_shape[d];

                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    // TODO: we use inplace_annotation to implement the reference_tensor, i.e., the
                    // output 0 shares the same address with input 0
                    // need to add a new annotation type or ref_tensor mechanism in the future
                    ctx->annotations->add_in_place_oi_pair({0, 0, false});
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& cfg = generic_op->localOpConfig.getRoot();
                    std::vector<int64_t> indices = cfg["indices"];
                    std::string Tindices =
                        cfg["Tindices"].is_null() ? "int32_t" : (std::string)cfg["Tindices"];

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "const " << Tindices << " indices[] = {" << join(indices) << "};\n";
                    lu << "const uint32_t ref_offset = " << ref_offset << ";\n";
                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if(tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        lu << "uint32_t offset = indices[tid/ref_offset] * ref_offset  + tid % "
                              "ref_offset;\n";
                        lu << "atomic_" << name << "(input1 + offset, input2[tid]);\n";
                        lu << "input1[offset] = input1[offset] * " << momentum
                           << " + input2[tid];\n";
                        if (use_nesterov)
                            lu << "atomic_" << string(CudaOpMap<nnfusion::op::Subtract>::op)
                               << "(input0 + offset, input2[tid] * " << lr << " + input1[offset] * "
                               << momentum << " * " << lr << ");\n";
                        else
                            lu << "atomic_" << string(CudaOpMap<nnfusion::op::Subtract>::op)
                               << "(input0 + offset, input1[offset] * " << lr << ");\n";
                    }
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(
                        get_atomic_math_kernel(CudaOpMap<nnfusion::op::Subtract>::op,
                                               CudaOpMap<nnfusion::op::Subtract>::math_kernel,
                                               dtype.c_type_string()));
                    _lu->require(get_atomic_math_kernel(name, math_kernel, dtype.c_type_string()));
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 512;
                    size_t block_cnt = align_to_block_size(threads, block_size_x);

                    m_gridDim = dim3(block_cnt, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "SparseApplyMomentum",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::SparseApplyMomentum)
