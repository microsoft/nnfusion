#pragma once
#include "cuda_emitter.hpp"
#include "cuda_langunit.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            static inline dim3 maxdim3(dim3 lhs, dim3 rhs)
            {
                return dim3(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
            }
            class ControlFlowEmitter : public CudaEmitter
            {
            public:
                using CudaEmitter::CudaEmitter;
                LanguageUnit_p emit_device_function_body() override;

            protected:
                static std::pair<cuda::dim3, cuda::dim3>
                    get_subgraph_launch_config(const ir::Program& program);

                static std::map<std::string, int> get_subgraph_inputs(const ir::Program& program);

                static std::vector<ir::Instruction::Pointer>
                    get_fused_kernel(const ir::Program& program);

                void allocate_shared_memory(LanguageUnit_p _lu);

                void create_param_map(
                    const ir::Program& program,
                    const std::unordered_map<std::string, int>& subgraph_output_map);

                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor);
                std::string get_launch_bound(nnfusion::ir::Instruction::Pointer ins);
                size_t get_subgraph_shared_memory(const ir::Program& program);
                size_t m_shared_memory_size = 0;
                bool is_emitting_block_kernel = false;
                descriptor::Tensor::Pointer m_workspace;
                std::unordered_map<nnfusion::descriptor::Tensor::Pointer, std::string> m_param_map;
            };
        }
    }
}
