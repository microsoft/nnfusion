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
            class Tile : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;
                nnfusion::element::Type dtype;
                nnfusion::Shape strides, reduced_strides, in_shape, out_shape;
                vector<int> stride_magic, stride_shift, reduced_magic, reduced_shift;

            public:
                Tile(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    out_shape = m_context->outputs.front()->get_shape();
                    in_shape = m_context->inputs.front()->get_shape();
                    threads = ctx->outputs.front()->size(false);
                    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
                    strides = nnfusion::row_major_strides(out_shape);

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

                    reduced_strides = nnfusion::row_major_strides(in_shape);

                    for (int i = 0; i < in_shape.size(); i++)
                    {
                        int magic;
                        int shift;
                        std::tie(magic, shift) = idiv_magic_u64(in_shape[i]);
                        reduced_magic.push_back(magic);
                        reduced_shift.push_back(shift);
                    }
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << nnfusion::op::expand_vector("strides", strides, "size_t")
                       << nnfusion::op::expand_vector("stride_magic", stride_magic, "int")
                       << nnfusion::op::expand_vector("stride_shift", stride_shift, "int")
                       << nnfusion::op::expand_vector("reduced_magic", reduced_magic, "int")
                       << nnfusion::op::expand_vector("reduced_shift", reduced_shift, "int")
                       << nnfusion::op::expand_vector("reduced_strides", reduced_strides, "size_t")
                       << nnfusion::op::expand_vector("reduced_shape", in_shape, "size_t") << "\n";

                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if(tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        coordinate_transform_to_multi_d(lu,
                                                        "strides",
                                                        "stride_magic",
                                                        "stride_shift",
                                                        "tid",
                                                        "coordinate",
                                                        out_shape.size(),
                                                        true);

                        // index into reduced tensor from coordinates of non-reduced tensor
                        lu << "uint32_t reduced_idx = 0;\n";
                        for (size_t i = 0; i < out_shape.size(); i++)
                        {
                            lu << "coordinate" << i << " -= reduced_shape" << i
                               << " * division_by_invariant_multiplication(coordinate" << i
                               << ", reduced_magic" << i << ", reduced_shift" << i << ");\n";
                            lu << "reduced_idx += coordinate" << i << " * "
                               << "reduced_strides" << i << ";\n";
                        }
                        lu << "output0[tid] = load(input0, reduced_idx);";
                    }
                    lu.block_end();
                    return _lu;
                }

                virtual LanguageUnit_p get_division_by_invariant_multiplication()
                {
                    return declaration::division_by_invariant_multiplication;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(get_division_by_invariant_multiplication());
                    _lu->require(declaration::load);
                    return _lu;
                }

                void set_launch_config() override
                {
                    size_t nthreads = shape_size(out_shape);
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x =
                        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);
                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };

            class RocmTile : public Tile
            {
            public:
                RocmTile(shared_ptr<KernelContext> ctx)
                    : Tile(ctx)
                {
                }

                virtual LanguageUnit_p get_division_by_invariant_multiplication() override
                {
                    return declaration::rocm_division_by_invariant_multiplication;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Tile",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::Tile)

REGISTER_KERNEL_EMITTER("Tile",                                                    //op_name
                        Device(ROCM_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::RocmTile)                                            // constructor