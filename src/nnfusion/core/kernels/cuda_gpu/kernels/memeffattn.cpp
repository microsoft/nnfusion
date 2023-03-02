// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "memeffattn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::MemEffAttn::MemEffAttn(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
    , generic_op(static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
{
    auto v_tensor = m_context->inputs[2];
    auto v_shape = v_tensor->get_shape();
    auto& cfg = generic_op->localOpConfig.getRoot();
    num_heads = cfg["num_heads"];
    batch_size = cfg["batch_size"];
    seq_len = cfg["seq_len"];
    seq_len_kv = cfg["seq_len_kv"];
    head_size = cfg["head_size"];
    head_size_v = cfg["head_size_v"];
    p_dropout = cfg["p_dropout"];
    // softmax_scale = cfg["softmax_scale"];
    softmax_scale = (float)std::pow(head_size, -0.5);
    is_causal = cfg["is_causal"];
    workspace_tensor = allocate_tensor(v_shape, element::f32);
    bool kIs64x64 = head_size <= 64;
    bool kSingleValueIteration = head_size <= 128;
    if (kIs64x64 && kSingleValueIteration)
        idx = 1;
    else if (kIs64x64 && !kSingleValueIteration)
        idx = 2;
    else if (!kIs64x64 && kSingleValueIteration)
        idx = 3;
    else
        idx = 4;
}

LanguageUnit_p cuda::MemEffAttn::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto code = nnfusion::op::create_code_from_template(
        R"( 
int64_t batch_size {@batch_size@};  
mem_eff_attention_@idx@(
    output0,
    input0, input1, input2,
    reinterpret_cast<float*>(@workspace_ptr@),
    &batch_size,
    @seq_len@,
    @seq_len_kv@,
    @num_heads@,
    @head_size@,
    @head_size_v@,
    @p_dropout@,
    @softmax_scale@,
    @is_causal@, stream
    );
    )",
        {{"seq_len", seq_len},
         {"seq_len_kv", seq_len_kv},
         {"batch_size", batch_size},
         {"num_heads", num_heads},
         {"head_size", head_size},
         {"head_size_v", head_size_v},
         {"p_dropout", 0}, //
         {"softmax_scale", softmax_scale},
         {"is_causal", is_causal},
         {"idx", idx},
         {"workspace_ptr", workspace_tensor->get_name()}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::MemEffAttn::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::mem_eff_attn);
    _lu->require(header::cutlass);
    _lu->require(header::kernel_forward);
    _lu->require(header::cuda_fp16);
    _lu->require(header::iostream);
    return _lu;
}

LanguageUnit_p cuda::MemEffAttn::emit_function_signature()
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
        // default name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void "
       << "(cudaStream_t stream, " << join(params, ", ") << ")";
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "MemEffAttn",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cutlass").Priority(9), // attrs
    cuda::MemEffAttn)                                                         // constructor

REGISTER_KERNEL_EMITTER(
    "MemEffAttn",                                                             // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cutlass").Priority(9), // attrs
    cuda::MemEffAttn)
