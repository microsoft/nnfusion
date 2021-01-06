// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "embed_layer_norm.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::EmbedLayerNorm::EmbedLayerNorm(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
    , generic_op(static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
{
    // input: input_ids, segment_ids, word_embedding, position_embedding, segment_embedding,
    //        gamma, beta, mask
    auto input_ids_tensor = m_context->inputs[0];
    auto input_ids_shape = input_ids_tensor->get_shape();
    auto word_embedding_tensor = m_context->inputs[2];
    auto word_embedding_shape = word_embedding_tensor->get_shape();
    dtype = word_embedding_tensor->get_element_type();
    batch_size = input_ids_shape[0];
    sequence_length = input_ids_shape[1];
    hidden_size = word_embedding_shape[1];

    auto& cfg = generic_op->localOpConfig.getRoot();
    epsilon = cfg["epsilon"];
    NNFUSION_CHECK(epsilon >= 0);
}

LanguageUnit_p cuda::EmbedLayerNorm::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    if (m_context->inputs.size() > 7)
    {
        lu << "ComputeMaskIndex(stream, " << sequence_length << ", " << batch_size
           << ", input7, static_cast<int*>(output1));\n";
    }

    if (dtype == element::f16)
    {
        lu << "EmbedSkipLayerNorm<half>(stream, " << hidden_size << ", " << batch_size << ", "
           << sequence_length
           << ", input0, input1, reinterpret_cast<const half*>(input6), reinterpret_cast<const "
              "half*>(input5),reinterpret_cast<const half*>(input2), reinterpret_cast<const "
              "half*>(input3), reinterpret_cast<const half*>(input4), __float2half_rn("
           << epsilon << "), reinterpret_cast<half*>(output0));\n ";
    }
    else
    {
        lu << "EmbedSkipLayerNorm<float>(stream, " << hidden_size << ", " << batch_size << ", "
           << sequence_length
           << ", input0, input1, reinterpret_cast<const float*>(input6), reinterpret_cast<const "
              "float*>(input5), reinterpret_cast<const float*>(input2), reinterpret_cast<const "
              "float*>(input3), reinterpret_cast<const float*>(input4), "
           << epsilon << ", reinterpret_cast<float*>(output0));\n ";
    }
    return _lu;
}

LanguageUnit_p cuda::EmbedLayerNorm::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cub);
    _lu->require(declaration::embed_skip_layer_norm);
    if (m_context->inputs.size() > 7)
        _lu->require(declaration::compute_mask_index);

    return _lu;
}

LanguageUnit_p cuda::EmbedLayerNorm::emit_function_signature()
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

REGISTER_KERNEL_EMITTER(
    "EmbedLayerNorm",                                                          // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::EmbedLayerNorm)                                                      // constructor

REGISTER_KERNEL_EMITTER(
    "EmbedLayerNorm",                                                          // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cuda_lib").Priority(2), // attrs
    cuda::EmbedLayerNorm)
