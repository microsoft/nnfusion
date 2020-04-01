// Microsoft (c) 2019, NNFusion Team

#include "gather_1d.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Gather1D::Gather1D(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto gather = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
    input_shape_0 = nnfusion::Shape(ctx->inputs[0]->get_shape());
    input_shape_1 = nnfusion::Shape(ctx->inputs[1]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

    axis = gather->localOpConfig.getRoot()["axis"];
    if (axis < 0)
    {
        axis = input_shape_0.size() + axis;
    }
    CHECK(axis < input_shape_0.size());

    int64_t outer_size = 1;
    int64_t inner_size = 1;
    for (int i = 0; i < axis; i++)
    {
        outer_size *= input_shape_0[i];
    }
    for (int i = axis + 1; i < input_shape_0.size(); i++)
    {
        inner_size *= input_shape_0[i];
    }

    int64_t out_size = nnfusion::shape_size(output_shape);
    CHECK(out_size > 0);
    is_axis_zero = outer_size == 1;
    gather_dim_size = input_shape_0[axis];
    indices_size = nnfusion::shape_size(input_shape_1);
    slice_size = inner_size;

    std::stringstream tag;
    tag << join(input_shape_0, "_") << join(input_shape_1, "_") << join(output_shape, "_")
        << "_axis" << axis;
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Gather1D::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << m_context->dtypes[0] << "* params = input0;\n";
    lu << m_context->dtypes[1] << "* indices = input1;\n";
    lu << m_context->dtypes[2] << "* out = output0;\n";

    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (i < " << nthreads << ")\n";
    lu.block_begin();
    {
        lu << "uint32_t batch_i = 0;\n"
           << "uint32_t indices_i = 0;\n"
           << "uint32_t slice_i = 0;\n";
        if (is_axis_zero)
        {
            lu << "indices_i = i / " << slice_size << ";\n"
               << "slice_i = i - indices_i * " << slice_size << ";\n";
        }
        else
        {
            lu << "uint32_t batch_indices_i = i / " << slice_size << ";\n"
               << "batch_i = batch_indices_i / " << indices_size << ";\n"
               << "indices_i = batch_indices_i - batch_i * " << indices_size << ";\n"
               << "slice_i = i - batch_indices_i * " << slice_size << ";\n";
        }

        lu << "uint32_t gather_i = __ldg(indices + indices_i);\n";
        lu << "if (gather_i >= " << gather_dim_size << ")\n"
           << "   out[i] = 0;\n"
           << "else\n";
        lu.block_begin();
        {
            lu << "uint32_t params_i = (batch_i * " << gather_dim_size << " + gather_i) * "
               << slice_size << " + slice_i;\n"
               << "out[i] = __ldg(params + params_i);\n";
        }
        lu.block_end();
    }
    lu.block_end();

    return _lu;
}

void cuda::Gather1D::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Gather1D::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}
REGISTER_KERNEL_EMITTER("GatherV2",                                                   // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::Gather1D)                                               // constructor