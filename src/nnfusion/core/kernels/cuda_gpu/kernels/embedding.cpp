// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "embedding.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::EmbeddingGrad::EmbeddingGrad(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto embedding_grad = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
    indices_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    y_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    x_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
}

LanguageUnit_p cuda::EmbeddingGrad::emit_function_signature()
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

LanguageUnit_p cuda::EmbeddingGrad::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    int64_t num_indices = shape_size(indices_shape);
    int64_t stride = y_shape[y_shape.size() - 1];
    int64_t num_weights = shape_size(x_shape) / stride;

    auto code = nnfusion::op::create_code_from_template(
        R"(
embedding_dense_backward_cuda<float, float, int64_t, float>(
stream, output0, @num_indices@, @stride@, input1, input0, @num_weights@, @padding_idx@);
    )",
        {
            {"num_indices", num_indices},
            {"stride", stride},
            {"num_weights", num_weights},
            {"padding_idx", 0},
        });

    lu << code << "\n";
    return _lu;
}

LanguageUnit_p cuda::EmbeddingGrad::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cub);
    LanguageUnit_p pt_embedding_grad(new LanguageUnit("declaration::ort_embed_skip_layer_norm"));
    *pt_embedding_grad << R"(
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

static const int EMB_BWD_BLOCKDIMY = 32;
static const int EMB_BWD_CUDA_WARP_SIZE = 32;
constexpr int EMB_BWD_NROWS_PER_THREAD = 10;
constexpr int EMB_BWD_MAX_BLOCK_SIZE = 1024;

template <typename scalar_t>
__global__ void arange_fill(scalar_t* __restrict__ output, const size_t range)
{
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < range)
    {
        output[tid] = tid;
    }
}

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below(the embedding_backward_feature_kernel function)
// is mostly copied from Pytorch aten/src/ATen/native/cuda/Embedding.cu

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void embedding_backward_feature_kernel(
    const index_t* indices,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ grad_weight,
    const int n, // OK to pass as int, we don't expect 2 billion+ samples in one shot
    const int64_t stride,
    const int padding_idx)
{
    extern __shared__ char buf[];
    accscalar_t* smem = (accscalar_t*)buf;
    accscalar_t* my_s = smem + EMB_BWD_CUDA_WARP_SIZE * threadIdx.y;
    int* indices_batch = (int*)(buf + sizeof(accscalar_t) * EMB_BWD_CUDA_WARP_SIZE * blockDim.y);

    const int s = (int)stride; // OK to make int, we don't expect 2 billion+ embedding row size

    const int f = threadIdx.x + blockIdx.x * blockDim.x; // feature_dim

    for (int batch_start = 0; batch_start < n; batch_start += blockDim.x * blockDim.y)
    {
        // Entire block cooperates to load a batch of 1024 indices to process
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        if (batch_start + tid < n)
            indices_batch[tid] = (int)indices[batch_start + tid];

        int batch_end =
            batch_start + blockDim.x * blockDim.y < n ? batch_start + blockDim.x * blockDim.y : n;

        // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
        for (int chunk_start = batch_start; chunk_start < batch_end; chunk_start += blockDim.y)
        {
            // This does double duty:  it makes sure indices_batch is ready, and it makes sure match-group
            // leaders are done with their accumulates before other warps start loading again.
            __syncthreads();

            int n_this_chunk =
                (batch_end - chunk_start) < blockDim.y ? (batch_end - chunk_start) : blockDim.y;

            int src_row = chunk_start + threadIdx.y;
            int dst_row =
                indices_batch[src_row - batch_start]; // This warp's target row in grad_weight

            // All warps load their smem segments with incoming grad data
            if (src_row < n && f < s && dst_row != padding_idx)
                my_s[threadIdx.x] = static_cast<accscalar_t>(grad[src_row * stride + f]);

            __syncthreads();

            // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
            // We need to check if any other warps pulled grad data targeting dst_row.
            // If so, we elect the first warp in each matching group as the leader.
            // Each leader warp serializes the accumulates targeting dst_row in shared memory,
            // then finishes by adding the accumulated buffer to dst_row in grad_weight.
            if (dst_row != padding_idx &&
                src_row < n) // Per-warp exit condition, safe with ballot_sync
            {
                int match_found_this_thread =
                    (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
                if (threadIdx.x >= n_this_chunk)
                    match_found_this_thread = 0;

                unsigned int matchmask = __ballot_sync(0xffffffff, match_found_this_thread);
                int first_remaining_peer = __ffs(matchmask) - 1;

                if (threadIdx.y ==
                    first_remaining_peer) // Nominate lowest-indexed warp as the leader
                {
                    matchmask ^= (1 << first_remaining_peer);
                    while (matchmask)
                    {
                        first_remaining_peer = __ffs(matchmask) - 1;
                        my_s[threadIdx.x] +=
                            smem[threadIdx.x + EMB_BWD_CUDA_WARP_SIZE * first_remaining_peer];
                        matchmask ^= (1 << first_remaining_peer);
                    }
                    if (f < s)
                        grad_weight[dst_row * stride + f] +=
                            static_cast<scalar_t>(my_s[threadIdx.x]);
                }
            }
        }
    }
}

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below(from the ceil_div function to the sum_and_scatter function)
// is mostly copied from Pytorch aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu

/* This code computes the sum of the weights in two-steps:
  1) Each GPU warp sums `EMB_BWD_NROWS_PER_THREAD` number of row given by `indeces`
  2) Each partial-sum from 1) are summed and scatter into `grad_weight`

  Notice, `EMB_BWD_NROWS_PER_THREAD` impacts the Achieved Occupancy of the
  kernel execution. If it is high, the size of the thread blocks will be
  too small to achieve good occupancy. Similarly, a very low value will
  make the size of the thread blocks in the final sum in step 2) too small.
*/

// Fast ceil division (no overflow checking)
__host__ __device__ __forceinline__ int64_t ceil_div(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

template <typename index_t>
__global__ void krn_partials_per_segment(index_t* ret,
                                         const index_t* segment_offsets,
                                         int64_t num_of_segments,
                                         int64_t numel)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_of_segments)
    {
        const int64_t idx_start = segment_offsets[id];
        const int64_t idx_end = (id == num_of_segments - 1) ? numel : segment_offsets[id + 1];
        const int64_t size = idx_end - idx_start;
        ret[id] = ceil_div(size, EMB_BWD_NROWS_PER_THREAD);
    }
}

template <typename index_t>
__global__ void krn_partial_segment_offset(index_t* ret,
                                           const index_t* partials_per_segment,
                                           const index_t* partials_per_segment_offset,
                                           const index_t* segment_offsets,
                                           int64_t num_of_segments)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_of_segments)
    {
        index_t idx = partials_per_segment_offset[id];
        const index_t num_partials = partials_per_segment[id];
        const index_t segment_offset = segment_offsets[id];
        for (int64_t i = 0; i < num_partials; ++i)
        {
            ret[idx++] = segment_offset + i * EMB_BWD_NROWS_PER_THREAD;
        }
    }
}

template <typename scalar_t, typename index_t>
__global__ void compute_grad_weight(const index_t* indices,
                                    const scalar_t* gradOutput,
                                    index_t* count,
                                    int64_t numel,
                                    const int64_t stride,
                                    const index_t* segment_offsets,
                                    const int64_t num_of_segments,
                                    scalar_t* grad_weight_per_segment,
                                    const int64_t stride_warped)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int id = gid / stride_warped;
    const int startFeature = gid % stride_warped;
    if (startFeature >= stride)
    {
        return;
    }
    if (id >= num_of_segments)
    {
        return;
    }
    const int idx_begin = segment_offsets[id];
    const int idx_end = (id == num_of_segments - 1) ? numel : segment_offsets[id + 1];

    scalar_t weight = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx)
    {
        const index_t target_row = indices[idx];
        const scalar_t scale = count ? (scalar_t)1.0 / count[idx] : 1.0;
        weight += gradOutput[target_row * stride + startFeature] * scale;
    }
    grad_weight_per_segment[id * stride + startFeature] = weight;
}

// This kernel assumes that all input tensors are contiguous.
template <typename scalar_t, typename index_t>
__global__ void sum_and_scatter(const index_t* input,
                                scalar_t* gradWeight,
                                const int64_t stride,
                                const index_t* segment_offsets,
                                const int64_t num_of_segments,
                                const scalar_t* grad_weight_per_segment,
                                const index_t* segment_sizes_offsets,
                                const int64_t num_of_partial_segments,
                                const int64_t padding_idx,
                                const int64_t stride_warped)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int id = gid / stride_warped;
    const int startFeature = gid % stride_warped;
    if (startFeature >= stride)
    {
        return;
    }
    if (id >= num_of_segments)
    {
        return;
    }

    const int idx_begin = segment_sizes_offsets[id];
    const int idx_end =
        (id == num_of_segments - 1) ? num_of_partial_segments : segment_sizes_offsets[id + 1];
    scalar_t weight = 0;
    for (int idx = idx_begin; idx < idx_end; ++idx)
    {
        weight += grad_weight_per_segment[idx * stride + startFeature];
    }
    int64_t target_row = input[segment_offsets[id]];
    if (target_row != padding_idx)
    {
        gradWeight[target_row * stride + startFeature] = weight;
    }
}

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below(the embedding_backward_cuda_kernel_unique_by_key function)
// is mostly copied from Pytorch aten/src/ATen/native/cuda/LegacyThrustHelpers.cu

template <typename index_t>
int64_t embedding_backward_cuda_kernel_unique_by_key(cudaStream_t stream,
                                                     const int64_t numel,
                                                     index_t* sorted_indices,
                                                     index_t* segment_offsets)
{
    auto sorted_indices_dev = thrust::device_ptr<index_t>(sorted_indices);
    static index_t* dummy = NULL;
    if (!dummy)
    {
        cudaMalloc(&dummy, numel * sizeof(index_t));
    }
    auto dummy_dev = thrust::device_ptr<index_t>(dummy);
    static index_t* range = NULL;
    if (!range)
    {
        cudaMalloc(&range, numel * sizeof(index_t));
    }
    arange_fill<index_t>
        <<<ceil_div(numel, EMB_BWD_MAX_BLOCK_SIZE), EMB_BWD_MAX_BLOCK_SIZE, 0, stream>>>(range,
                                                                                         numel);
    auto range_dev = thrust::device_ptr<index_t>(range);
    auto ends = thrust::unique_by_key_copy(thrust::device,
                                           sorted_indices_dev,
                                           sorted_indices_dev + numel,
                                           range_dev,
                                           dummy_dev,
                                           thrust::device_ptr<index_t>(segment_offsets));
    auto ret = thrust::get<0>(ends) - dummy_dev;
    return ret;
}

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below(the embedding_backward_cuda_kernel function)
// is mostly copied from Pytorch aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu

template <typename input_t, typename output_t, typename index_t, typename acc_t>
void embedding_backward_cuda_kernel(cudaStream_t stream,
                                    output_t* grad_weight,
                                    const int64_t numel,
                                    const int64_t stride,
                                    const input_t* grad,
                                    const index_t* orig_indices,
                                    index_t* sorted_indices,
                                    int64_t num_weights,
                                    int padding_idx,
                                    bool mode_mean = false)
{
    // Compute the number of segments and their start position so that we do not have to
    // spawn a warp per index. In this context, a segment is a number of rows that should
    // be summarized.
    // Unit: index in `sorted_indices` and `orig_indices`
    static index_t* segment_offsets = NULL;
    if (!segment_offsets)
    {
        cudaMalloc(&segment_offsets, numel * sizeof(index_t));
    }
    int64_t num_of_segments = embedding_backward_cuda_kernel_unique_by_key<index_t>(
        stream, numel, sorted_indices, segment_offsets);

    // We split the segments up into sizes of `EMB_BWD_NROWS_PER_THREAD`
    // Compute the number partial-segments per segment (some partial-segments
    // may not be the full `EMB_BWD_NROWS_PER_THREAD` number of rows)
    static index_t* partials_per_segment = NULL;
    if (!partials_per_segment)
    {
        // cudaMalloc(&partials_per_segment, num_of_segments * sizeof(index_t));
        cudaMalloc(&partials_per_segment, numel * sizeof(index_t)); // malloc upper bound
    }
    {
        krn_partials_per_segment<<<ceil_div(num_of_segments, 32), 32, 0, stream>>>(
            partials_per_segment, segment_offsets, num_of_segments, numel);
    }

    // In order to compute `partial_segment_offset`, which is the start index
    // of each partial-segment in `sorted_indices`, we need to compute the
    // start position of each _segment_ in `partial_segment_offset`.
    // Unit: index in `partial_segment_offset`
    static index_t* partials_per_segment_offset = NULL;
    if (!partials_per_segment_offset)
    {
        // cudaMalloc(&partials_per_segment_offset, num_of_segments * sizeof(index_t));
        cudaMalloc(&partials_per_segment_offset, numel * sizeof(index_t)); // malloc upper bound
    }
    static void* d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0;
    if (!d_temp_storage)
    {
        cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      partials_per_segment,
                                      partials_per_segment_offset,
                                      num_of_segments,
                                      stream,
                                      false);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  partials_per_segment,
                                  partials_per_segment_offset,
                                  num_of_segments,
                                  stream,
                                  false);

    // The total number of partial-segments is the sum of `partials_per_segment_offset`
    index_t partials_per_segment_ns_1, partials_per_segment_offset_ns_1;
    cudaMemcpy(&partials_per_segment_ns_1,
               partials_per_segment + num_of_segments - 1,
               sizeof(index_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&partials_per_segment_offset_ns_1,
               partials_per_segment_offset + num_of_segments - 1,
               sizeof(index_t),
               cudaMemcpyDeviceToHost);
    const int num_of_partial_segments =
        (int)(partials_per_segment_ns_1 + partials_per_segment_offset_ns_1);

    // Now we can compute the start position of each partial-segment
    // Unit: index in `sorted_indices` and `orig_indices`
    static index_t* partial_segment_offset = NULL;
    if (!partial_segment_offset)
    {
        // cudaMalloc(&partial_segment_offset, num_of_partial_segments * sizeof(index_t));
        cudaMalloc(&partial_segment_offset, numel * sizeof(index_t)); // malloc upper bound
    }
    {
        krn_partial_segment_offset<<<ceil_div(num_of_segments, 32), 32, 0, stream>>>(
            partial_segment_offset,
            partials_per_segment,
            partials_per_segment_offset,
            segment_offsets,
            num_of_segments);
    }

    const int stride_warped = ceil_div(stride, EMB_BWD_CUDA_WARP_SIZE) * EMB_BWD_CUDA_WARP_SIZE;
    const int block = std::min(stride_warped, EMB_BWD_MAX_BLOCK_SIZE);
    const int grid = ceil_div(num_of_partial_segments * stride_warped, block);

    // For numerical stability, the dtype of `grad_weight_per_segment`
    // should match `acc_type`
    static input_t* grad_weight_per_segment = NULL;
    if (!grad_weight_per_segment)
    {
        // cudaMalloc(&grad_weight_per_segment, num_of_partial_segments * stride * sizeof(input_t));
        cudaMalloc(&grad_weight_per_segment, numel * stride * sizeof(input_t));
    }
    // Compute the sum of each partial-segment and handle bags
    compute_grad_weight<input_t, index_t>
        <<<grid, block, 0, stream>>>(orig_indices,
                                     grad,
                                     nullptr,
                                     numel,
                                     stride,
                                     partial_segment_offset,
                                     (int64_t)num_of_partial_segments,
                                     grad_weight_per_segment,
                                     stride_warped);

    // Finally, we sum all the partial-sums and scatter them
    // into `grad_weight`.
    const int grid2 = ceil_div(num_of_segments * stride_warped, block);
    sum_and_scatter<input_t, index_t><<<grid2, block, 0, stream>>>(sorted_indices,
                                                                   grad_weight,
                                                                   stride,
                                                                   segment_offsets,
                                                                   num_of_segments,
                                                                   grad_weight_per_segment,
                                                                   partials_per_segment_offset,
                                                                   num_of_partial_segments,
                                                                   padding_idx,
                                                                   stride_warped);
}

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below(the embedding_dense_backward_cuda function)
// is mostly copied from Pytorch aten/src/ATen/native/cuda/Embedding.cu

template <typename input_t, typename output_t, typename index_t, typename acc_t>
void embedding_dense_backward_cuda(cudaStream_t stream,
                                   output_t* grad_weight,
                                   const int64_t num_indices,
                                   const int64_t stride,
                                   const input_t* grad,
                                   const index_t* indices,
                                   int64_t num_weights,
                                   int64_t padding_idx)
{
    if (num_indices <= 3072)
    {
        dim3 grid(ceil_div(stride, (int64_t)EMB_BWD_CUDA_WARP_SIZE));
        dim3 block(EMB_BWD_CUDA_WARP_SIZE, EMB_BWD_BLOCKDIMY);

        embedding_backward_feature_kernel<input_t, acc_t, index_t>
            <<<grid,
               block,
               sizeof(acc_t) * EMB_BWD_CUDA_WARP_SIZE * EMB_BWD_BLOCKDIMY +
                   sizeof(int) * EMB_BWD_CUDA_WARP_SIZE * EMB_BWD_BLOCKDIMY,
               stream>>>(indices,
                         grad,
                         grad_weight,
                         static_cast<int>(num_indices),
                         static_cast<int64_t>(stride),
                         static_cast<int>(padding_idx));
        return;
    }

    static index_t* sorted_indices = NULL;
    if (!sorted_indices)
    {
        cudaMalloc(&sorted_indices, num_indices * sizeof(index_t));
    }
    static index_t* orig_indices = NULL;
    if (!orig_indices)
    {
        cudaMalloc(&orig_indices, num_indices * sizeof(index_t));
    }
    static index_t* range = NULL;
    if (!range)
    {
        cudaMalloc(&range, num_indices * sizeof(index_t));
    }
    arange_fill<index_t>
        <<<ceil_div(num_indices, EMB_BWD_MAX_BLOCK_SIZE), EMB_BWD_MAX_BLOCK_SIZE, 0, stream>>>(
            range, num_indices);
    static void* d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0;
    if (!d_temp_storage)
    {
        cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                        temp_storage_bytes,
                                        indices,
                                        sorted_indices,
                                        range,
                                        orig_indices,
                                        num_indices,
                                        0,
                                        sizeof(index_t) * 8,
                                        stream,
                                        false);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }
    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                    temp_storage_bytes,
                                    indices,
                                    sorted_indices,
                                    range,
                                    orig_indices,
                                    num_indices,
                                    0,
                                    sizeof(index_t) * 8,
                                    stream,
                                    false);

    embedding_backward_cuda_kernel<input_t, output_t, index_t, acc_t>(stream,
                                                                      grad_weight,
                                                                      num_indices,
                                                                      stride,
                                                                      grad,
                                                                      orig_indices,
                                                                      sorted_indices,
                                                                      num_weights,
                                                                      padding_idx,
                                                                      false);
}
)";
    _lu->require(pt_embedding_grad);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "EmbeddingGrad",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::EmbeddingGrad)                                                       // constructor
