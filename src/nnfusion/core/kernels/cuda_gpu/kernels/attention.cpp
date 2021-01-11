// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attention.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Attention::Attention(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
    , generic_op(static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
{
    // input: input, weight, bias, mask_index(optional), past(optional)
    auto input_tensor = m_context->inputs[0];
    auto input_shape = input_tensor->get_shape();

    dtype = input_tensor->get_element_type();
    batch_size = input_shape[0];
    sequence_length = input_shape[1];
    hidden_size = input_shape[2];

    auto& cfg = generic_op->localOpConfig.getRoot();
    num_heads = cfg["num_heads"];
    unidirectional = cfg["unidirectional"];
    head_size = hidden_size / num_heads;

    gemm_tensor = allocate_tensor(Shape({batch_size * sequence_length, 3 * hidden_size}), dtype);
    ones_tensor = allocate_tensor(Shape({batch_size * sequence_length}), dtype);

    if (m_context->inputs.size() > 3)
    {
        if (m_context->inputs[3]->get_shape()[0] > batch_size)
        {
            mask_start = "input3 + " + to_string(batch_size);
        }
    }

    if (m_context->inputs.size() > 4)
    {
        past_sequence_length = m_context->inputs[4]->get_shape()[3];
    }

    use_2d_attention_mask =
        (m_context->inputs.size() == 5 && m_context->inputs[3]->get_shape().size() == 2);
    if (use_2d_attention_mask || unidirectional)
        NNFUSION_CHECK(sequence_length + past_sequence_length <= 1024)
            << "Attention CUDA operator does not supported 2D attention mask or unidirectional "
               "with total sequence length > 1024.";
    size_t len =
        batch_size * num_heads * sequence_length * (sequence_length + past_sequence_length);
    size_t bytes = len * dtype.size();
    size_t alignment = 256;
    size_t bytesAligned = ((bytes + alignment - 1) / alignment) * alignment;
    workspace_size =
        3 * batch_size * sequence_length * num_heads * head_size * dtype.size() + 2 * bytesAligned;
}

LanguageUnit_p cuda::Attention::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto code = nnfusion::op::create_code_from_template(
        R"(
        const @dtype@ alpha = 1.0f;
        const @dtype@ beta = 0.f;
        CUBLAS_SAFE_CALL(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        CUBLAS_SAFE_CALL(@cublasGemm@(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, @n@, @m@, 1, &alpha, 
        reinterpret_cast<const @dtype@*>(input2), @n@, reinterpret_cast<const @dtype@*>(@ones_tensor@), 
        1, &beta, @gemm_tensor@, @n@));

        void *workspace_ptr = NULL;
        CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, @workspace_size@));
        cudaStream_t stream = nullptr;
        CUBLAS_SAFE_CALL(cublasGetStream(cublas_handle, &stream));
        QkvToContext<@dtype@>(cublas_handle, stream,
                        @batch_size@, @sequence_length@, @num_heads@, @head_size@, @element_size@,
                        reinterpret_cast<@dtype@*>(input0), reinterpret_cast<@dtype@*>(output0), reinterpret_cast<@dtype@*>(workspace_ptr),
                        @input3@, @is_unidirectional@,
                        @past_sequence_length@, @input4@, @output1@, @use_2d_attention_mask@, @mask_start@);

    )",
        {{"n", 3 * hidden_size},
         {"m", batch_size * sequence_length},
         {"dtype", (dtype == element::f16) ? "half" : "float"},
         {"cublasGemm", (dtype == element::f16) ? "cublasHgemm" : "cublasSgemm"},
         {"ones_tensor", ones_tensor->get_name()},
         {"gemm_tensor", gemm_tensor->get_name()},
         //{"workspace_tensor", workspace_tensor->get_name()},
         {"unidirectional", unidirectional},
         {"batch_size", batch_size},
         {"sequence_length", sequence_length},
         {"num_heads", num_heads},
         {"head_size", head_size},
         {"element_size", dtype.size()},
         {"past_sequence_length", past_sequence_length},
         {"is_unidirectional", unidirectional},
         {"input3", (m_context->inputs.size() >= 4) ? "reinterpret_cast<const int*>(input3)" : "nullptr"},
         {"input4", (m_context->inputs.size() == 5) ? (dtype == element::f16) ? "reinterpret_cast<half*>(input4)" : "reinterpret_cast<float*>(input4)" : "nullptr"},
         {"present", (m_context->outputs.size() > 1) ? "reinterpret_cast<const int*>(output1)" : "nullptr"},
         {"workspace_size", workspace_size},
         {"use_2d_attention_mask", use_2d_attention_mask},
         {"mask_start", mask_start}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::Attention::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::ort_qkv_to_context);
    _lu->require(header::cublas);
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
