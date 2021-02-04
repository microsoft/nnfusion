// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "skip_layer_norm.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::SkipLayerNorm::SkipLayerNorm(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
    , generic_op(static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
{
    // input: input, skip, gama, beta, bias(optional)
    auto input_tensor = m_context->inputs[0];
    auto input_shape = input_tensor->get_shape();
    element_count = shape_size(input_shape);
    dtype = input_tensor->get_element_type();
    sequence_length = input_shape[1];
    hidden_size = input_shape[2];
    auto& cfg = generic_op->localOpConfig.getRoot();
    epsilon = cfg["epsilon"];
    NNFUSION_CHECK(epsilon >= 0);
}

LanguageUnit_p cuda::SkipLayerNorm::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto code = nnfusion::op::create_code_from_template(
        R"(
ComputeSkipLayerNorm<@dtype@>(
stream,
@hidden_size@,
@element_count@,
reinterpret_cast<const @dtype@*>(input0),
reinterpret_cast<const @dtype@*>(input1),
reinterpret_cast<const @dtype@*>(input3),
reinterpret_cast<const @dtype@*>(input2),
reinterpret_cast<const @dtype@*>(@input4@),
@expression1@@epsilon@@expression2@,
reinterpret_cast<@dtype@*>(output0));
    )",
        {{"hidden_size", hidden_size},
         {"element_count", element_count},
         {"dtype", (dtype == element::f16) ? "half" : "float"},
         {"input4", (m_context->inputs.size() == 5) ? "input4" : "nullptr"},
         {"expression1", (dtype == element::f16) ? "__float2half_rn(" : ""},
         {"expression2", (dtype == element::f16) ? ")" : ""},
         {"epsilon", epsilon}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::SkipLayerNorm::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cub);
    LanguageUnit_p ort_skip_layer_norm(new LanguageUnit("declaration::ort_skip_layer_norm"));
    *ort_skip_layer_norm << R"(
template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernelSmall(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias, 
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;

  if (threadIdx.x < ld) {
    val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[threadIdx.x];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  LayerNormSmall<T, TPB>(val, thread_data, ld, idx, beta, gamma, epsilon, output);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(
    const int ld, const T* input, const T* skip, const T* beta, const T* gamma, const T* bias, 
    const T epsilon, T* output) {
  const T reverse_ld = T(1.f / ld);
  const int offset = blockIdx.x * ld;

  KeyValuePairSum pair_sum;
  // reduce x and x^2
  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = (bias == nullptr) ? input[idx] + skip[idx] : input[idx] + skip[idx] + bias[i];
    const T rldval = reverse_ld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
    output[idx] = val;
  }

  LayerNorm<T, TPB>(thread_data, ld, offset, beta, gamma, epsilon, output);
}

template <typename T>
void ComputeSkipLayerNorm(
    cudaStream_t stream, const int ld, const int n, const T* input, const T* skip,
    const T* beta, const T* gamma, const T* bias, const T epsilon, T* output) {
  // this must be true because n is the total size of the tensor
  assert(n % ld == 0);
  const int grid_size = n / ld;

  if (ld <= 32) {
    constexpr int block_size = 32;
    SkipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    SkipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else if (ld == 384) {
    constexpr int block_size = 384;
    SkipLayerNormKernelSmall<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  } else {
    constexpr int block_size = 256;
    SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(ld, input, skip, beta, gamma, bias, epsilon, output);
  }
}
    )";
    ort_skip_layer_norm->require(declaration::ort_layer_norm);
    declaration::ort_layer_norm->require(declaration::math_Rsqrt);
    _lu->require(ort_skip_layer_norm);
    return _lu;
}

LanguageUnit_p cuda::SkipLayerNorm::emit_function_signature()
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
    "SkipLayerNorm",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::SkipLayerNorm)                                                       // constructor

REGISTER_KERNEL_EMITTER(
    "SkipLayerNorm",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cuda_lib").Priority(2), // attrs
    cuda::SkipLayerNorm)
