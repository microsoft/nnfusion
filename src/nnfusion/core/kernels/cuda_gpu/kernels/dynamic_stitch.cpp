// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dynamic_stitch.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::DynamicStitch::DynamicStitch(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto dynamic_stitch_node =
        static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());

    std::vector<std::vector<int32_t>> indices_inputs =
        dynamic_stitch_node->localOpConfig.getRoot()["indices_inputs"];
    num_partitions = dynamic_stitch_node->localOpConfig.getRoot()["N"];
    NNFUSION_CHECK(num_partitions == indices_inputs.size());

    auto indice0_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    auto data0_shape = nnfusion::Shape(ctx->inputs[num_partitions]->get_shape());
    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    output_size = shape_size(output_shape);

    int32_t max_index = -1;
    int first_dim_size;
    int data_elements_size = 0;

    for (auto& indices : indices_inputs)
    {
        for (auto v : indices)
        {
            max_index = std::max(v, max_index);
        }
        data_elements_size += indices.size();
    }

    first_dim_size = max_index + 1;
    NNFUSION_CHECK(first_dim_size == output_shape[0])
        << "first_dim_size=" << first_dim_size << ", output_shape=[" << join(output_shape, ",")
        << "]";

    slice_size = 1;
    for (int d = indice0_shape.size(); d < data0_shape.size(); d++)
    {
        slice_size *= data0_shape[d];
    }

    indices_flat.resize(first_dim_size, -1);
    data_flat.resize(data_elements_size, "");

    // data_flat index
    int32_t idx = 0;
    // sum of indices_inputs[i].NumElements() for compute indicies_flat value.
    int32_t base_size = 0;

    for (int i = 0; i < indices_inputs.size(); ++i)
    {
        auto& indices_vec = indices_inputs[i];
        std::string data_ptr_base("input" + std::to_string(i + num_partitions));
        for (int j = 0; j < indices_vec.size(); ++j)
        {
            // indices_flat's indices represent the indices of output.
            // indices_flat's values represent the indices of input_data where the
            // data located.
            indices_flat[indices_vec[j]] = base_size + j;

            data_flat[idx] = data_ptr_base + " + " + to_string(j * slice_size);
            ++idx;
        }
        base_size += indices_vec.size();
    }
}

LanguageUnit_p cuda::DynamicStitch::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu.block_begin();
    {
        lu << "const int32_t data_indices[] = {" << join(indices_flat) << "};\n";
        lu << "const " << m_context->dtypes[num_partitions] << "* data_ptrs[] = {"
           << join(data_flat) << "};\n";

        lu << "for (uint32_t output_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
           << "     output_index < " << output_size << ";\n"
           << "     output_index += gridDim.x * blockDim.x)\n"
           << "{\n"
           << "  const int32_t slice_id = output_index / " << slice_size << ";\n"
           << "  const int32_t slice_offset = output_index % " << slice_size << ";\n"
           << "  const int32_t input_index = data_indices[slice_id];\n"
           << "  if (input_index != -1)\n"
           << "  {\n"
           << "    output0[output_index] = __ldg(data_ptrs[input_index] + slice_offset);\n"
           << "  }\n"
           << "}\n";
    }
    lu.block_end();

    return _lu;
}

void cuda::DynamicStitch::set_launch_config()
{
    uint32_t nthreads = output_size;
    uint32_t block_size_x = 1024;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::DynamicStitch::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "DynamicStitch",                                                              // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::DynamicStitch)                                                          // constructor