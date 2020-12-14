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
            class ReverseSequence : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                nnfusion::Shape strides;
                Shape out_shape;
                uint32_t seq_axis, max_seq_len, batch_axis, threads;
                vector<int> stride_magic, stride_shift;

            public:
                ReverseSequence(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    seq_axis = generic_op->localOpConfig.get("seq_axis");
                    batch_axis = generic_op->localOpConfig.get("batch_axis");
                    out_shape = ctx->outputs[0]->get_shape();
                    threads = shape_size(out_shape);

                    // calculate strides
                    strides = nnfusion::row_major_strides(out_shape);
                    // precacluate invariants for integer division via multiplication
                    for (int i = 0; i < strides.size(); i++)
                    {
                        int magic;
                        int shift;
                        std::tie(magic, shift) = idiv_magic_u64(strides[i]);
                        stride_magic.push_back(magic);
                        stride_shift.push_back(shift);
                    }
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit calc_cood;
                    coordinate_transform_to_multi_d(calc_cood,
                                                    "strides",
                                                    "stride_magic",
                                                    "stride_shift",
                                                    "tid",
                                                    "coordinate",
                                                    out_shape.size(),
                                                    true);
                    auto code = nnfusion::op::create_code_from_template(
                        R"(uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
@strides@
@strides_magic@
@strides_shift@
if (tid < @threads@) {
    @calc_cood@
    int seq_dim = coordinate@seq_id@;
    int batch_dim = coordinate@batch_id@;
    int new_dim = input1[batch_dim] - 1 - seq_dim;
    int from_tid = new_dim >= 0 ? tid + (new_dim - seq_dim) * @seq_stride@ : tid;
    output0[tid] = input0[from_tid];
})",
                        {{"strides", nnfusion::op::expand_vector("strides", strides, "size_t")},
                         {"strides_magic",
                          nnfusion::op::expand_vector("stride_magic", stride_magic, "int")},
                         {"strides_shift",
                          nnfusion::op::expand_vector("stride_shift", stride_shift, "int")},
                         {"threads", threads},
                         {"calc_cood", calc_cood.get_code()},
                         {"seq_id", seq_axis},
                         {"batch_id", batch_axis},
                         {"seq_stride", strides[seq_axis]}});
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name(), code));
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(get_division_by_invariant_multiplication());
                    return _lu;
                }

                virtual LanguageUnit_p get_division_by_invariant_multiplication()
                {
                    return declaration::division_by_invariant_multiplication;
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

            class RocmReverseSequence : public ReverseSequence
            {
            public:
                RocmReverseSequence(shared_ptr<KernelContext> ctx)
                    : ReverseSequence(ctx)
                {
                }

                virtual LanguageUnit_p get_division_by_invariant_multiplication() override
                {
                    return declaration::rocm_division_by_invariant_multiplication;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER(
    "ReverseSequence",                                                            // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::ReverseSequence)                                                        // constructor

REGISTER_KERNEL_EMITTER("ReverseSequence",                                         // op_name
                        Device(ROCM_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::RocmReverseSequence)                                 // constructor