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
    //        gamma, beta, mask(optional)
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

    auto code = nnfusion::op::create_code_from_template(
        R"(
EmbedSkipLayerNorm<@dtype@>(
stream, @hidden_size@, @batch_size@, @sequence_length@, input0, input1,
reinterpret_cast<const @dtype@*>(input6), reinterpret_cast<const @dtype@*>(input5),
reinterpret_cast<const @dtype@*>(input2), reinterpret_cast<const @dtype@*>(input3), 
reinterpret_cast<const @dtype@*>(input4), @expression1@@epsilon@@expression2@,
reinterpret_cast<@dtype@*>(output0));
    )",
        {{"hidden_size", hidden_size},
         {"batch_size", batch_size},
         {"sequence_length", sequence_length},
         {"dtype", (dtype == element::f16) ? "half" : "float"},
         {"expression1", (dtype == element::f16) ? "__float2half_rn(" : ""},
         {"expression2", (dtype == element::f16) ? ")" : ""},
         {"epsilon", epsilon}});

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::EmbedLayerNorm::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cub);
    LanguageUnit_p ort_embed_layer_norm(new LanguageUnit("declaration::ort_embed_skip_layer_norm"));
    *ort_embed_layer_norm << R"(
template <typename T, unsigned TPB>
__global__ void EmbedLayerNormKernel(
    int hidden_size, const int* input_ids, const int* segment_ids, const T* beta, const T* gamma,
    const T* word_embedding, const T* position_embedding, const T* segment_embedding,
    const T epsilon, T* output) {
  KeyValuePairSum pair_sum;
  // 1. lookup word and segment of the block
  // blockIdx.x = position in the sequence
  // blockIdx.y = batch
  // gridDim.x = sequence_length
  // gridDim.y = batch_size
  __shared__ int word_id;
  __shared__ int segment_id;

  const T rld = T(1.f / hidden_size);
  const int sequence_position = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    word_id = input_ids[sequence_position];
    segment_id = segment_ids[sequence_position];
  }
  __syncthreads();

  // 2. load pos/segment/word embeddings and add them toghether
  // offset into embeddings is given by word_id * hidden_size
  const int position_offset = blockIdx.x * hidden_size;
  const int word_offset = word_id * hidden_size;
  const int segment_offset = segment_id * hidden_size;
  // the output offset is given by b * (sequence_length * hidden_size) + s * hidden_size
  const int output_offset = sequence_position * hidden_size;

  cub::KeyValuePair<T, T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden_size; it += TPB) {
    const T w(word_embedding[word_offset + it]);
    const T t(segment_embedding[segment_offset + it]);
    const T p(position_embedding[position_offset + it]);
    const T val = w + t + p;

    output[output_offset + it] = val;
    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, cub::KeyValuePair<T, T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  LayerNorm<T, TPB>(thread_data, hidden_size, output_offset, beta, gamma, epsilon, output);
}

template <typename T>
void EmbedSkipLayerNorm(
    cudaStream_t stream, int hidden_size, int batch_size, int sequence_length,
    const int* input_ids, const int* segment_ids, const T* beta, const T* gamma,
    const T* word_embedding, const T* position_embedding, const T* segment_embedding,
    const T epsilon, T* output) {
  constexpr int tpb = 256;
  const dim3 grid(sequence_length, batch_size, 1);
  const dim3 block(tpb, 1, 1);

  EmbedLayerNormKernel<T, tpb>
      <<<grid, block, 0, stream>>>(hidden_size, input_ids, segment_ids, beta, gamma, word_embedding, position_embedding, segment_embedding, epsilon, output);
}
    )";
    ort_embed_layer_norm->require(declaration::ort_layer_norm);
    _lu->require(ort_embed_layer_norm);

    LanguageUnit_p ort_compute_mask_index(new LanguageUnit("declaration::ort_compute_mask_index"));
    *ort_compute_mask_index << R"(
template <unsigned TPB>
__global__ void MaskIndexKernelSmall(int sequence_length, const int* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  const int idx = offset + threadIdx.x;
  if (threadIdx.x < sequence_length) {
    const int val = mask[idx];
    if (val == 0)  // masked position: report thread idx
    {
      thread_data = threadIdx.x;
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

template <unsigned TPB>
__global__ void MaskIndexKernel(int sequence_length, const int* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  for (int i = threadIdx.x; i < sequence_length; i += TPB) {
    const int idx = offset + i;
    const int val = mask[idx];
    if (val == 0)  // masked position: report thread idx
    {
      thread_data = min(thread_data, i);
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

inline void ComputeMaskIndex(cudaStream_t stream, const int sequence_length, const int batch_size, const int* mask, int* mask_index) {
  // Mask idx is of length batch_size and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  // Assume n = batch_size x sequence_length
  if (sequence_length <= 32) {
    MaskIndexKernelSmall<32><<<batch_size, 32, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 128) {
    MaskIndexKernelSmall<128><<<batch_size, 128, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length == 384) {
    MaskIndexKernelSmall<384><<<batch_size, 384, 0, stream>>>(sequence_length, mask, mask_index);
  } else {
    MaskIndexKernel<256><<<batch_size, 256, 0, stream>>>(sequence_length, mask, mask_index);
  }
} 
    )";
    if (m_context->inputs.size() > 7)
        _lu->require(ort_compute_mask_index);
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
