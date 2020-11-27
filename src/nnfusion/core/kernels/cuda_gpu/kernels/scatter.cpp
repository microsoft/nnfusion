// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class Scatter : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads, ref_offset, indicies_count;
                nnfusion::element::Type dtype;
                nnfusion::descriptor::Tensor::Pointer t_data, t_update, t_indices;

            public:
                Scatter(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    NNFUSION_CHECK_NOT_NULLPTR(CudaOpMap<T>::atomic)
                        << "No Scatter operator for this OP.";

                    t_data = m_context->inputs[0];
                    t_indices = m_context->inputs[1];
                    t_update = m_context->inputs[2];

                    threads = shape_size(t_update->get_shape());
                    dtype = nnfusion::element::Type(t_data->get_element_type());

                    ref_offset = 1;
                    // Row major
                    for (size_t i = 1; i < t_data->get_shape().size(); i++)
                        ref_offset *= t_data->get_shape()[i];

                    indicies_count = shape_size(t_indices->get_shape());

                    //\todo(wenxh): This must be applied and Add a new field to replace "destructive = False".
                    if (!ctx->annotations)
                    {
                        ctx->annotations = std::make_shared<Annotations>();
                        ctx->annotations->add_in_place_oi_pair({0, 0, false});
                    }
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
    const uint32_t ref_offset = @ref_offset@;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < @threads@)
    {
        uint32_t offset = input1[tid/ref_offset] * ref_offset  + tid % ref_offset;
        // Output0 is reference to input0;
        atomic_@atomic_op@(input0 + offset, input2[tid]);
    }
)",
                        {{"ref_offset", ref_offset},
                         {"threads", threads},
                         {"atomic_op", string(CudaOpMap<T>::op)}});
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name(), code));
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(get_atomic_math_kernel(
                        CudaOpMap<T>::op, CudaOpMap<T>::math_kernel, dtype.c_type_string()));
                    _lu->require(declaration::load);
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x =
                        align_to_block_size(static_cast<uint32_t>(threads), block_size_x);
                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_SCATTER_KERNEL(OP_NAME, KERNEL_NAME)                                              \
    REGISTER_KERNEL_EMITTER(                                                                       \
        "" #KERNEL_NAME "",                                                                        \
        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("scatter").Priority(2),                  \
        cuda::Scatter<nnfusion::op::OP_NAME>);

REGISTER_SCATTER_KERNEL(Subtract, ScatterSub);
REGISTER_SCATTER_KERNEL(Add, ScatterAdd);
REGISTER_SCATTER_KERNEL(Minimum, ScatterMin);
REGISTER_SCATTER_KERNEL(Maximum, ScatterMax);
// REGISTER_SCATTER_KERNEL(And);
// REGISTER_SCATTER_KERNEL(Or);