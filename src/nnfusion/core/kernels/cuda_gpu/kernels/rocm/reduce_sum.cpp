// Microsoft (c) 2019, NNFusion Team

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class RocmReduce : public CudaEmitter
            {
            public:
                RocmReduce(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& ctx = m_context;
                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    size_t in_size = 1LU, out_size = 1LU;
                    for (auto it : input_shape)
                        in_size *= it;
                    for (auto it : output_shape)
                        out_size *= it;

                    size_t blockY;
                    std::string lda_offset;
                    if (out_size == 1)
                    {
                        blockY = 1;
                        lda_offset = "";
                    }
                    else if (input_shape.size() == 2 && output_shape.size() == 1 &&
                             input_shape[0] != input_shape[1] && input_shape[0] == output_shape[0])
                    {
                        blockY = input_shape[0];
                        lda_offset = "blockIdx.y * dataSize + ";
                        // cancel fixing this branch
                        return nullptr;
                    }
                    else
                        return nullptr;

                    m_gridDim = dim3(1, blockY, 1);
                    m_blockDim = dim3(64, 1, 1);

                    auto templ = nnfusion::op::create_code_from_template(
                        R"(
    constexpr unsigned int gridDimX = 1;
    constexpr unsigned int blockSize = 64;
    constexpr unsigned int gridDimY = @gridDimY@;
    constexpr unsigned int dataSize = @dataSize@;

    extern __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    sdata[tid] = 0;
    #pragma unroll
    while (i < dataSize) { sdata[tid] += input0[@lda_offset@i]; i += blockSize * gridDimX; }
    if (blockSize >= 128) { __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }

    // warp_reduce
    volatile float *__sdata = (volatile float *)sdata;
    if (blockSize >= 128) __sdata[tid] += __sdata[tid + 64];
#ifndef __HIP_PLATFORM_HCC__
    __syncthreads();
#endif
    if (blockSize >= 64) __sdata[tid] += __sdata[tid + 32];
    if (blockSize >= 32) __sdata[tid] += __sdata[tid + 16];
    if (blockSize >= 16) __sdata[tid] += __sdata[tid + 8];
    if (blockSize >= 8) __sdata[tid] += __sdata[tid + 4];
    if (blockSize >= 4) __sdata[tid] += __sdata[tid + 2];
    if (blockSize >= 2) __sdata[tid] += __sdata[tid + 1];

    if (tid == 0) output0[blockIdx.y] = sdata[0];
                        )",
                        {{"gridDimY", m_gridDim.y},
                         {"dataSize", in_size},
                         {"lda_offset", lda_offset}});

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << templ;
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
// Microsoft (c) 2019, NNFusion Team

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_GPU_KERNEL(KEY, OP_NAME)                                                          \
    REGISTER_KERNEL_EMITTER(KEY,                                                                   \
                            Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("PRIORITY_2" #OP_NAME),  \
                            cuda::RocmReduce<nnfusion::op::OP_NAME>)

REGISTER_GPU_KERNEL("Sum", Add)
