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

            protected:
                static std::pair<cuda::dim3, cuda::dim3>
                    get_subgraph_launch_config(const ir::Program& program);

                static std::map<std::string, int> get_subgraph_inputs(const ir::Program& program);

                static std::vector<ir::Instruction::Pointer>
                    get_fused_kernel(const ir::Program& program);

                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor);
                std::string get_launch_bound(nnfusion::ir::Instruction::Pointer ins);
            };
        }
    }
}
