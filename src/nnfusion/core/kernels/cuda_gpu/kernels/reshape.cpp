// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reshape.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Reshape::Reshape(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    NNFUSION_CHECK(ctx->outputs[0]->size(false) > 0) << "Invalid output shape for Reshape.";
    reshape = static_pointer_cast<nnfusion::op::Reshape>(ctx->gnode->get_op_ptr());
    is_memcpy = false;
    is_noop = false;
    //Noop
    if (ctx->outputs[0]->get_name() == ctx->inputs[0]->get_name())
    {
        is_noop = true;
        // NNFUSION_LOG(INFO) << "Same input and output tensor.";
        return;
    }

    arg_shape = ctx->inputs[0]->get_shape();
    arg_rank = arg_shape.size();
    result_shape = ctx->outputs[0]->get_shape();
    input_order = reshape->get_input_order();
    size_t result_shape_product = shape_size(result_shape);

    //Result OP
    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    if (!reshape->get_is_layout_change() || result_shape_product < 2)
    {
        is_memcpy = true;
        // NNFUSION_LOG(INFO) << "No need for zero-size or 1-d tensor reshape.";
        // add inplace tag
        if (!ctx->annotations)
            ctx->annotations = std::make_shared<Annotations>();
        ctx->annotations->add_in_place_oi_pair({0, 0, false});
        return;
    }

    //combine inordered dimensons after reorder in shape, update output shape and input order
    Shape in_order_map(arg_rank, 0);
    for (int i = 0; i < arg_rank - 1; i++)
    {
        if (static_cast<int64_t>(input_order[i + 1]) - static_cast<int64_t>(input_order[i]) == 1)
        {
            in_order_map[input_order[i]] = 1;
        }
    }

    Shape combine_arg_shape;
    Shape combine_idx_map(arg_rank, 0);
    Shape combine_input_order;
    size_t shape_i = 1;
    size_t combine_rank = 0;
    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[i] == 1)
        {
            shape_i *= arg_shape[i];
        }
        else
        {
            combine_arg_shape.push_back(shape_i * arg_shape[i]);
            shape_i = 1;
            combine_idx_map[i] = combine_rank++;
        }
    }

    for (int i = 0; i < arg_rank; i++)
    {
        if (in_order_map[input_order[i]] == 0)
        {
            combine_input_order.push_back(combine_idx_map[input_order[i]]);
        }
    }

    //eleminate dimenson size = 1, update input order and output shape
    Shape new_arg_shape;
    Shape new_result_shape;
    Shape new_idx_map(combine_rank, 0);
    Shape new_input_order;
    size_t new_rank = 0;

    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[i] != 1)
        {
            new_arg_shape.push_back(combine_arg_shape[i]);
            new_idx_map[i] = new_rank++;
        }
    }
    for (int i = 0; i < combine_rank; i++)
    {
        if (combine_arg_shape[combine_input_order[i]] != 1)
        {
            new_input_order.push_back(new_idx_map[combine_input_order[i]]);
        }
    }
    for (int i = 0; i < new_rank; i++)
    {
        new_result_shape.push_back(new_arg_shape[new_input_order[i]]);
    }

    arg_shape = new_arg_shape;
    arg_rank = arg_shape.size();
    result_shape = new_result_shape;
    input_order = new_input_order;
}

LanguageUnit_p cuda::Reshape::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

cuda::Reshape2D::Reshape2D(shared_ptr<KernelContext> ctx)
    : Reshape(ctx)
{
    // <TODO> currently we set it to 16, will add tuning method later
    block_size = 16;
    input_strides = row_major_strides(arg_shape);
    output_strides = nnfusion::NVShape(arg_rank);
    trans_strides = nnfusion::NVShape(arg_rank);
    int stride = 1;
    for (int64_t i = arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= arg_shape[input_order[i]];
    }
    for (int64_t i = 0; i < arg_rank; i++)
    {
        trans_strides[input_order[i]] = output_strides[i];
    }

    std::stringstream tag;
    tag << "cuda_reshape_2D"
        << "_i_" << join(arg_shape, "_") << "_o_" << join(input_order, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Reshape2D::emit_function_body()
{
    if (is_noop || is_memcpy || arg_rank != 2)
    {
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto& data_type = m_context->dtypes[1];
    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[2]* output0)
    //lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t nx = " << arg_shape[1] << ";\n";
        lu << "size_t ny = " << arg_shape[0] << ";\n";
        // Common data area ends

        lu << "__shared__ " << data_type << " tile[" << block_size << "][" << block_size + 1
           << "];\n";
        lu << "uint32_t base1 = blockIdx.x * blockDim.x;\n";
        lu << "uint32_t base0 = blockIdx.y * blockDim.y;\n";
        lu << "uint32_t tid1 = threadIdx.x;\n";
        lu << "uint32_t tid0 = threadIdx.y;\n";
        lu << "uint32_t idx1 = base1 + tid1;\n";
        lu << "uint32_t idx0 = base0 + tid0;\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "input_idx += input_strides" << i << "* idx" << i << ";\n";
            }
            lu << "tile[tid0][tid1] = input0[input_idx];\n";
        }
        lu.block_end();

        lu << "idx1 = base1 + tid0;\n";
        lu << "idx0 = base0 + tid1;\n";
        lu << "__syncthreads();\n";

        lu << "if (idx1 < nx && idx0 < ny)\n";
        lu.block_begin();
        {
            lu << "uint32_t output_idx = 0;\n";
            for (int i = 0; i < 2; i++)
            {
                lu << "output_idx += trans_strides" << i << "* idx" << i << ";\n";
            }
            lu << "output0[output_idx] = tile[tid1][tid0];\n";
        }
        lu.block_end();
    }
    //lu.block_end();

    return _lu;
}

void cuda::Reshape2D::set_launch_config()
{
    if (is_noop || is_memcpy || arg_rank != 2)
    {
        return;
    }
    uint32_t aligned_grid_size_x = align_to_block_size(arg_shape[1], block_size);
    uint32_t aligned_grid_size_y = align_to_block_size(arg_shape[0], block_size);

    m_gridDim = dim3(aligned_grid_size_x, aligned_grid_size_y, 1);
    m_blockDim = dim3(block_size, block_size, 1);
}

cuda::Reshape3D::Reshape3D(shared_ptr<KernelContext> ctx)
    : Reshape(ctx)
{
    block_size = std::vector<uint32_t>(3, 0);
    // TODO: currently we set it to 16, will add tuning method later
    block_size_x = 16;
    block_size[0] = block_size_x;                                                        //x
    block_size[2] = (input_order.size() >= 3 && input_order[2] == 0) ? block_size_x : 1; //z
    block_size[1] = (block_size[2] == block_size_x) ? 1 : block_size_x;                  //y
    input_strides = nnfusion::row_major_strides(arg_shape);
    output_strides = nnfusion::NVShape(arg_rank);
    trans_strides = nnfusion::NVShape(arg_rank);
    int stride = 1;
    for (int64_t i = arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= arg_shape[input_order[i]];
    }
    for (int64_t i = 0; i < arg_rank; i++)
    {
        trans_strides[input_order[i]] = output_strides[i];
    }

    std::stringstream tag;
    tag << "cuda_reshape_3D"
        << "_i_" << join(arg_shape, "_") << "_o_" << join(input_order, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Reshape3D::emit_function_body()
{
    if (is_noop || is_memcpy || arg_rank != 3)
    {
        return nullptr;
    }
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto& data_type = m_context->dtypes[1];
    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[2]* output0)
    //lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t nx = " << arg_shape[2] << ";\n";
        lu << "size_t ny = " << arg_shape[1] << ";\n";
        lu << "size_t nz = " << arg_shape[0] << ";\n";
        // Common data area ends

        lu << "__shared__ " << data_type << " tile[" << block_size[2] << "][" << block_size[1]
           << "][" << block_size[0] + 1 << "];\n";
        lu << "uint32_t base2 = blockIdx.x * blockDim.x;\n";
        lu << "uint32_t base1 = blockIdx.y * blockDim.y;\n";
        lu << "uint32_t base0 = blockIdx.z * blockDim.z;\n";
        lu << "uint32_t tid2 = threadIdx.x;\n";
        lu << "uint32_t tid1 = threadIdx.y;\n";
        lu << "uint32_t tid0 = threadIdx.z;\n";
        lu << "uint32_t otid2 = tid2;\n";
        lu << "uint32_t otid1 = tid1;\n";
        lu << "uint32_t otid0 = tid0;\n";
        lu << "uint32_t idx2 = base2 + tid2;\n";
        lu << "uint32_t idx1 = base1 + tid1;\n";
        lu << "uint32_t idx0 = base0 + tid0;\n";

        lu << "if (idx2 < nx && idx1 < ny && idx0 < nz)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = 0;\n";
            for (int i = 0; i < 3; i++)
            {
                lu << "input_idx += input_strides" << i << "* idx" << i << ";\n";
            }
            lu << "tile[tid0][tid1][tid2] = input0[input_idx];\n";
        }
        lu.block_end();

        if (input_order[2] == 1)
        {
            lu << "otid2 = tid1;\n";
            lu << "otid1 = tid2;\n";
        }
        else if (input_order[2] == 0)
        {
            lu << "otid2 = tid0;\n";
            lu << "otid0 = tid2;\n";
        }
        lu << "idx2 = base2 + otid2;\n";
        lu << "idx1 = base1 + otid1;\n";
        lu << "idx0 = base0 + otid0;\n";
        lu << "__syncthreads();\n";

        lu << "if (idx2 < nx && idx1 < ny && idx0 < nz)\n";
        lu.block_begin();
        {
            lu << "uint32_t output_idx = 0;\n";
            for (int i = 0; i < 3; i++)
            {
                lu << "output_idx += trans_strides" << i << "* idx" << i << ";\n";
            }
            lu << "output0[output_idx] = tile[otid0][otid1][otid2];\n";
        }
        lu.block_end();
    }
    //lu.block_end();

    return _lu;
}

void cuda::Reshape3D::set_launch_config()
{
    if (is_noop || is_memcpy || arg_rank != 3)
    {
        return;
    }
    uint32_t aligned_grid_size_x = align_to_block_size(arg_shape[2], block_size[0]);
    uint32_t aligned_grid_size_y = align_to_block_size(arg_shape[1], block_size[1]);
    uint32_t aligned_grid_size_z = align_to_block_size(arg_shape[0], block_size[2]);

    m_gridDim = dim3(aligned_grid_size_x, aligned_grid_size_y, aligned_grid_size_z);
    m_blockDim = dim3(block_size[0], block_size[1], block_size[2]);
}

cuda::ReshapehD::ReshapehD(shared_ptr<KernelContext> ctx)
    : Reshape(ctx)
{
    block_size_x = 64;
    input_strides = nnfusion::row_major_strides(arg_shape);
    output_strides = nnfusion::NVShape(arg_rank);
    trans_strides = nnfusion::NVShape(arg_rank);
    int stride = 1;
    for (int64_t i = arg_rank - 1; i >= 0; i--)
    {
        output_strides[i] = stride;
        stride *= arg_shape[input_order[i]];
    }
    for (int64_t i = 0; i < arg_rank; i++)
    {
        trans_strides[input_order[i]] = output_strides[i];
    }

    std::stringstream tag;
    tag << "cuda_reshape_D"
        << "_i_" << join(arg_shape, "_") << "_o_" << join(input_order, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::ReshapehD::emit_function_body()
{
    if (is_noop || is_memcpy || arg_rank == 3 || arg_rank == 2)
    {
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    uint32_t nthreads = static_cast<uint32_t>(shape_size(arg_shape));

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[2]* output0)
    //lu.block_begin();
    {
        // Common data area starts
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        lu << expand_vector_uint32("input_strides", input_strides);
        lu << expand_vector_uint32("trans_strides", trans_strides);
        lu << "size_t n = " << nthreads << ";\n";
        // Common data area ends

        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << "if (tid < n)\n";
        lu.block_begin();
        {
            lu << "uint32_t input_idx = tid;\n";
            lu << "uint32_t output_idx = 0;\n";
            size_t i = 0;
            for (; i < arg_rank - 1; i++)
            {
                lu << "output_idx += (input_idx / input_strides" << i << ") * trans_strides" << i
                   << ";\n";
                lu << "input_idx %= input_strides" << i << ";\n";
            }
            lu << "output_idx += (input_idx / input_strides" << i << ") * trans_strides" << i
               << ";\n";
            lu << "output0[output_idx] = input0[tid];\n";
        }
        lu.block_end();
    }
    //lu.block_end();

    return _lu;
}

void cuda::ReshapehD::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(arg_shape));
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

cuda::ReshapeMemcpy::ReshapeMemcpy(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    NNFUSION_CHECK(ctx->outputs[0]->size(false) > 0) << "Invalid output shape for Reshape.";
    reshape = static_pointer_cast<nnfusion::op::Reshape>(ctx->gnode->get_op_ptr());
    is_memcpy = false;
    is_noop = false;
    //Noop
    if (ctx->outputs[0]->get_name() == ctx->inputs[0]->get_name())
    {
        is_noop = true;
    }

    arg_shape = ctx->inputs[0]->get_shape();
    arg_rank = arg_shape.size();
    result_shape = ctx->outputs[0]->get_shape();
    size_t result_shape_product = shape_size(result_shape);

    //Result OP
    //for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
    if (!reshape->get_is_layout_change() || result_shape_product < 2)
    {
        is_memcpy = true;
        // NNFUSION_LOG(INFO) << "No need for zero-size or 1-d tensor reshape.";

        // add inplace tag
        if (!ctx->annotations)
            ctx->annotations = std::make_shared<Annotations>();
        ctx->annotations->add_in_place_oi_pair({0, 0, false});
    }

    std::stringstream tag;
    tag << "cuda_reshape_Memcpy"
        << "_i_" << join(arg_shape, "_") << "_o_" << join(input_order, "_");
    custom_tag = tag.str();
}

bool cuda::ReshapeMemcpy::is_eliminative()
{
    if (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]))
        return true;
    else
        return false;
}

LanguageUnit_p cuda::ReshapeMemcpy::emit_function_body()
{
    if (!is_memcpy && !is_noop)
    {
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    if (is_memcpy)
    {
        lu << "if (input0 != output0) {\n"
           << "   cudaMemcpyAsync(output0, input0, " << static_cast<uint32_t>(shape_size(arg_shape))
           << " * sizeof(" << m_context->dtypes[0] << ")"
           << ", cudaMemcpyDeviceToDevice, stream);\n"
           << "}\n";
    }
    else
    {
        lu << "// noop as input0 == output0.\n";
    }

    return _lu;
}

LanguageUnit_p cuda::ReshapeMemcpy::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

LanguageUnit_p cuda::ReshapeMemcpy::emit_function_signature()
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
        // defult name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void "
       << "(cudaStream_t stream, " << join(params, ", ") << ")";
    return _lu;
}

// Register Reshape kernel emitter

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                                       // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel_2D").Priority(2), // attrs
    cuda::Reshape2D)                                                                 // constructor

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                                       // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel_3D").Priority(2), // attrs
    cuda::Reshape3D)                                                                 // constructor

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                                      // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel_D").Priority(2), // attrs
    cuda::ReshapehD)                                                                // constructor

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                                 // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::ReshapeMemcpy)                                                       // constructor
