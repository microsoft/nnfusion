// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attention.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Attention::Attention(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
    , generic_op(static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
{
    // input: input, mask_index(optional), past(optional)
    auto input_tensor = m_context->inputs[0];
    auto input_shape = input_tensor->get_shape();

    dtype = input_tensor->get_element_type();
    auto& cfg = generic_op->localOpConfig.getRoot();
    num_heads = cfg["num_heads"];
    batch_size = cfg["batch_size"];
    sequence_length = cfg["sequence_length"];
    past_sequence_length = cfg["past_sequence_length"];
    head_size = cfg["head_size"];
    unidirectional = cfg["unidirectional"];
    hidden_size = num_heads * head_size;

    if (m_context->inputs.size() > 1)
    {
        if (m_context->inputs[1]->get_shape()[0] > batch_size)
        {
            mask_start = "reinterpret_cast<const int*>(input1 + " + to_string(batch_size) + ")";
        }
    }

    use_2d_attention_mask =
        (m_context->inputs.size() >= 2 && m_context->inputs[1]->get_shape().size() == 2);
    if (use_2d_attention_mask || unidirectional)
        NNFUSION_CHECK(sequence_length + past_sequence_length <= 1024)
            << "Attention CUDA operator does not supported 2D Attention mask or unidirectional "
               "with total sequence length > 1024.";
    size_t len =
        batch_size * num_heads * sequence_length * (sequence_length + past_sequence_length);
    size_t bytes = len * dtype.size();
    size_t alignment = 256;
    size_t bytesAligned = ((bytes + alignment - 1) / alignment) * alignment;
    workspace_size =
        3 * batch_size * sequence_length * num_heads * head_size * dtype.size() + 2 * bytesAligned;
    workspace_size = 3 * batch_size * sequence_length * num_heads * head_size + 2 * len;
    workspace_tensor = allocate_tensor(Shape({workspace_size}), dtype);
}

LanguageUnit_p cuda::Attention::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto code = nnfusion::op::create_code_from_template(
        R"(      
cudaStream_t stream = nullptr;
CUBLAS_SAFE_CALL(cublasGetStream(cublas_handle, &stream));
QkvToContext<@dtype@>(cublas_handle, stream,
                @batch_size@, @sequence_length@, @num_heads@, @head_size@, @element_size@,
                reinterpret_cast<@dtype@*>(input0), reinterpret_cast<@dtype@*>(output0), reinterpret_cast<@dtype@*>(@workspace_ptr@),
                @input1@, @is_unidirectional@,
                @past_sequence_length@, @input2@, @output1@, @use_2d_attention_mask@, @mask_start@);
    )",
        {{"n", 3 * hidden_size},
         {"m", batch_size * sequence_length},
         {"k", hidden_size},
         {"dtype", (dtype == element::f16) ? "half" : "float"},
         {"batch_size", batch_size},
         {"sequence_length", sequence_length},
         {"num_heads", num_heads},
         {"head_size", head_size},
         {"element_size", dtype.size()},
         {"past_sequence_length", past_sequence_length},
         {"is_unidirectional", unidirectional},
         {"input1",
          (m_context->inputs.size() >= 2) ? "reinterpret_cast<const int*>(input1)" : "nullptr"},
         {"input2",
          (m_context->inputs.size() == 3)
              ? (dtype == element::f16) ? "reinterpret_cast<half*>(input2)"
                                        : "reinterpret_cast<float*>(input2)"
              : "nullptr"},
         {"output1",
          (m_context->outputs.size() > 1) ? "reinterpret_cast<const int*>(output1)" : "nullptr"},
         {"use_2d_attention_mask", use_2d_attention_mask},
         {"mask_start", mask_start},
         {"workspace_ptr", workspace_tensor->get_name()}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::Attention::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::math_constants);
    _lu->require(declaration::ort_qkv_to_context);
    _lu->require(header::cublas);
    _lu->require(header::cub);
    _lu->require(macro::CUBLAS_SAFE_CALL);
    _lu->require(macro::CUDA_SAFE_CALL);
    return _lu;
}

LanguageUnit_p cuda::Attention::emit_function_signature()
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
       << "(cublasHandle_t cublas_handle, " << join(params, ", ") << ")";
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Attention",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cublas").Priority(2), // attrs
    cuda::Attention)                                                         // constructor

REGISTER_KERNEL_EMITTER(
    "Attention",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cublas").Priority(2), // attrs
    cuda::Attention)
