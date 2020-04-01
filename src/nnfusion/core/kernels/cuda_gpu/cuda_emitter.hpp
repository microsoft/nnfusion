// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "cuda_helper.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            struct dim3
            {
                dim3()
                    : x(1)
                    , y(1)
                    , z(1)
                {
                }
                dim3(int x, int y, int z)
                    : x(x)
                    , y(y)
                    , z(z)
                {
                }
                int x, y, z;
            };

            class CudaEmitter : public KernelEmitter
            {
            public:
                CudaEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda")
                {
                }

                virtual bool is_static_function() override { return false; }
                // Need to regenerate function call with new assigned launch config(stream).
                LanguageUnit_p emit_function_call() override;

            protected:
                // config the blockDim and gridDim
                virtual void set_launch_config() = 0;

                LanguageUnit_p emit_function_signature() override;

                dim3 m_blockDim;
                dim3 m_gridDim;
            };

            class BlockCudaEmitter : public CudaEmitter
            {
            public:
                BlockCudaEmitter(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , num_local_thread_sync(0)
                {
                }

                LanguageUnit_p emit_block_kernel(int block_id)
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel"));
                    auto& lu = *_lu;

                    int block_size = m_blockDim.x * m_blockDim.y * m_blockDim.z;
                    int block_num = m_gridDim.x * m_gridDim.y * m_gridDim.z;
                    // CHECK(block_id < block_num);

                    FunctionUnit_p fu = this->get_or_emit_source();
                    lu << fu->comment_unit->get_code();
                    lu << this->emit_device_function_signature()->get_code() << "\n";
                    lu.block_begin();
                    lu << "if (threadIdx.x >= " << block_size << ")";
                    lu.block_begin();
                    if (num_local_thread_sync > 0)
                    {
                        lu << "for (int i = 0; i < " << num_local_thread_sync
                           << "; i++) __syncthreads();\n";
                    }
                    lu << "return;\n";
                    lu.block_end();

                    lu << "const dim3 blockDim(" << m_blockDim.x << ", " << m_blockDim.y << ", "
                       << m_blockDim.z << ");\n";
                    lu << "const dim3 gridDim(" << m_gridDim.x << ", " << m_gridDim.y << ", "
                       << m_gridDim.z << ");\n";

                    if (m_blockDim.y != 1 && m_blockDim.z == 1)
                    {
                        lu << "const dim3 threadIdx(threadIdx.x % " << m_blockDim.x
                           << ", threadIdx.x / " << m_blockDim.x << ", 0);\n";
                    }
                    else if (m_blockDim.y != 1 && m_blockDim.z != 1)
                    {
                        lu << "const dim3 threadIdx(threadIdx.x % " << m_blockDim.x
                           << ", threadIdx.x / " << m_blockDim.x << " % " << m_blockDim.y
                           << ", threadIdx.x / " << m_blockDim.x * m_blockDim.y << ");\n";
                    }

                    if (m_gridDim.y == 1 && m_gridDim.z == 1)
                    {
                        lu << "const dim3 blockIdx(block_id, 0, 0);\n";
                    }
                    else if (m_gridDim.z == 1)
                    {
                        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / "
                           << m_gridDim.x << ", 0);\n";
                    }
                    else
                    {
                        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / "
                           << m_gridDim.x << " % " << m_gridDim.y << ", block_id / "
                           << m_gridDim.x * m_gridDim.y << ");\n";
                    }

                    lu << fu->body_unit->get_code() << "\n";
                    lu.block_end();

                    return _lu;
                }

                LanguageUnit_p emit_device_function_signature();

                // this API can only be used inner the function body
                void emit_thread_sync(LanguageUnit_p lu)
                {
                    *lu << "__syncthreads();\n";
                    num_local_thread_sync++;
                }

            private:
                size_t num_local_thread_sync;
            };

            class CudaElementwiseEmitter : public BlockCudaEmitter
            {
            public:
                CudaElementwiseEmitter(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                }

                virtual std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel() = 0;
            };

            class CudaLibEmitter : public KernelEmitter
            {
            public:
                CudaLibEmitter(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda_lib")
                {
                }
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion