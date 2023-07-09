// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// #include "../cuda_cudnn.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class GatherND : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                nnfusion::Shape input_shape, indices_shape, output_shape;
                int axis;

            public:
                GatherND(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                    generic_op =
                        static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    indices_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    axis = generic_op->localOpConfig.getRoot()["axis"];
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    auto input_size = shape_size(input_shape);
                    auto indices_size = shape_size(indices_shape);
                    auto output_size = shape_size(output_shape);

                    size_t batch_size = 1;
                    for (auto i = 0; i < axis; i++)
                    {
                        batch_size *= indices_shape[i];
                    }

                    size_t batch_input_size = input_size / batch_size;
                    size_t batch_indices_size = (indices_size / batch_size);
                    size_t batch_output_size = (output_size / batch_size);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << m_context->dtypes[0] << "* batch_input = input0;\n";
                    lu << m_context->dtypes[1] << "* batch_indices = input1;\n";
                    lu << m_context->dtypes[2] << "* batch_output = output0;\n";
                    lu << "uint32_t input_shape[] = {" << join(input_shape, ",") << "};\n";
                    lu << "uint32_t indices_shape[] = {" << join(indices_shape, ",") << "};\n";
                    lu << "uint32_t output_shape[] = {" << join(output_shape, ",") << "};\n";
                    lu << "uint32_t batch_dim = " << axis << ";\n";

                    uint32_t nthreads = static_cast<uint32_t>(output_size);
                    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (i < " << nthreads << ")\n";
                    lu.block_begin();
                    {
                        auto code = op::create_code_from_template(
                            R"(
uint32_t batch_size = @batch_size@;

uint32_t batch_input_size = @batch_input_size@;
uint32_t batch_indices_size = @batch_indices_size@;
uint32_t batch_output_size = @batch_output_size@;

uint32_t batch_index = i / batch_output_size;
uint32_t sub_index = i - batch_index * batch_output_size;

batch_input += batch_index * batch_input_size;
batch_indices += batch_index * batch_indices_size;
batch_output += batch_index * batch_output_size;

uint32_t current_output_size = batch_output_size;
uint32_t current_indices_size = batch_indices_size;
for (uint32_t j = batch_dim; j < @indices_shape_size@ - 1; j++)
{
    current_output_size /= output_shape[j];
    current_indices_size /= indices_shape[j];
    int sub_index_x = sub_index / current_output_size;
    sub_index -= sub_index_x * current_output_size;
    batch_output += sub_index_x * current_output_size;
    batch_indices += sub_index_x * current_indices_size;
}

uint32_t current_input_size = batch_input_size;
for (uint32_t j = 0; j < @indices_last_dim@; j++)
{
    current_input_size /= input_shape[batch_dim + j];
    batch_input += __ldg(batch_indices + j) * current_input_size;
}
batch_output[sub_index] = __ldg(batch_input + sub_index);
)",
                            {{"batch_size", batch_size},
                             {"batch_input_size", batch_input_size},
                             {"batch_indices_size", batch_indices_size},
                             {"batch_output_size", batch_output_size},
                             {"indices_shape_size", indices_shape.size()},
                             {"indices_last_dim", indices_shape.back()}});

                        lu << code;
                    }
                    lu.block_end();
                    return _lu;
                }

                void set_launch_config()
                {
                    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }
            };

            class GatherNDGrad : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                nnfusion::Shape x_shape, indices_shape, y_shape;
                int axis;

            public:
                GatherNDGrad(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                    generic_op =
                        static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());

                    indices_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    y_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
                    x_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    axis = generic_op->localOpConfig.getRoot()["axis"];
                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    ctx->annotations->add_in_place_oi_pair({0, 2, false});
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    auto input_size = shape_size(x_shape);
                    auto indices_size = shape_size(indices_shape);
                    auto output_size = shape_size(y_shape);

                    size_t batch_size = 1;
                    for (auto i = 0; i < axis; i++)
                    {
                        batch_size *= indices_shape[i];
                    }

                    size_t batch_input_size = input_size / batch_size;
                    size_t batch_indices_size = (indices_size / batch_size);
                    size_t batch_output_size = (output_size / batch_size);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << m_context->dtypes[0] << "* batch_indices = input0;\n";
                    lu << m_context->dtypes[1] << "* batch_input = input1;\n";
                    lu << m_context->dtypes[2] << "* batch_output = output0;\n";
                    lu << "uint32_t input_shape[] = {" << join(x_shape, ",") << "};\n";
                    lu << "uint32_t indices_shape[] = {" << join(indices_shape, ",") << "};\n";
                    lu << "uint32_t output_shape[] = {" << join(y_shape, ",") << "};\n";
                    lu << "uint32_t batch_dim = " << axis << ";\n";

                    uint32_t nthreads = static_cast<uint32_t>(output_size);
                    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (i < " << nthreads << ")\n";
                    lu.block_begin();
                    {
                        auto code = op::create_code_from_template(
                            R"(
uint32_t batch_size = @batch_size@;

uint32_t batch_input_size = @batch_input_size@;
uint32_t batch_indices_size = @batch_indices_size@;
uint32_t batch_output_size = @batch_output_size@;

uint32_t batch_index = i / batch_output_size;
uint32_t sub_index = i - batch_index * batch_output_size;

batch_input += batch_index * batch_input_size;
batch_indices += batch_index * batch_indices_size;
batch_output += batch_index * batch_output_size;

uint32_t current_output_size = batch_output_size;
uint32_t current_indices_size = batch_indices_size;
for (uint32_t j = batch_dim; j < @indices_shape_size@ - 1; j++)
{
    current_output_size /= output_shape[j];
    current_indices_size /= indices_shape[j];
    int sub_index_x = sub_index / current_output_size;
    sub_index -= sub_index_x * current_output_size;
    batch_output += sub_index_x * current_output_size;
    batch_indices += sub_index_x * current_indices_size;
}

uint32_t current_input_size = batch_input_size;
for (uint32_t j = 0; j < @indices_last_dim@; j++)
{
    current_input_size /= input_shape[batch_dim + j];
    batch_input += __ldg(batch_indices + j) * current_input_size;
}
uint32_t x_offset = batch_input + sub_index - input1;
uint32_t y_offset = batch_output + sub_index - output0;
atomic_add(output0 + x_offset, __ldg(input1 + y_offset));
)",
                            {{"batch_size", batch_size},
                             {"batch_input_size", batch_input_size},
                             {"batch_indices_size", batch_indices_size},
                             {"batch_output_size", batch_output_size},
                             {"indices_shape_size", indices_shape.size()},
                             {"indices_last_dim", indices_shape.back()}});

                        lu << code;
                    }
                    lu.block_end();
                    return _lu;
                }

                void set_launch_config()
                {
                    uint32_t nthreads =
                        static_cast<uint32_t>(std::max(shape_size(x_shape), shape_size(y_shape)));
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(get_atomic_math_kernel(CudaOpMap<op::Add>::op,
                                                        CudaOpMap<op::Add>::math_kernel,
                                                        m_context->dtypes[2]));
                    _lu->require(declaration::load);
                    return _lu;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register kernel emitter
using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("GatherND",                                                       // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"), // attrs
                        cuda::GatherND) // constructor

REGISTER_KERNEL_EMITTER("GatherNDGrad",                                                   // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"), // attrs
                        cuda::GatherNDGrad) // constructor
