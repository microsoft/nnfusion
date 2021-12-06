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
            class ControlFlowEmitter : public CudaEmitter
            {
            public:
                using CudaEmitter::CudaEmitter;

            protected:
                static inline cuda::dim3 maxdim3(cuda::dim3 lhs, cuda::dim3 rhs)
                {
                    return cuda::dim3(
                        std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
                }
                static std::pair<cuda::dim3, cuda::dim3>
                    get_subgraph_launch_config(const ir::Program& program)
                {
                    cuda::dim3 block_dim{0, 0, 0}, grid_dim{0, 0, 0};
                    for (auto blk : program)
                    {
                        for (auto ins : *blk)
                        {
                            auto kernel = ins->getKernel();
                            // neglect the Constant copy
                            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Result" ||
                                ins->getGNode()->get_op_type() == "Constant")
                                continue;
                            if (kernel->get_kernel_type() == "cuda")
                            {
                                auto cuda_kernel = static_pointer_cast<cuda::CudaEmitter>(kernel);
                                block_dim = maxdim3(block_dim, cuda_kernel->get_block_dim());
                                grid_dim = maxdim3(grid_dim, cuda_kernel->get_grid_dim());
                            }
                            else
                            {
                                NNFUSION_CHECK_FAIL();
                            }
                        }
                    }
                    return std::make_pair(block_dim, grid_dim);
                }

                static std::map<std::string, int> get_subgraph_inputs(const ir::Program& program)
                {
                    std::map<std::string, int> inputs;
                    int i = 0;
                    for (auto blk : program)
                        for (auto ins : *blk)
                        {
                            if (ins->getGNode()->get_op_type() == "Parameter" ||
                                ins->getGNode()->get_op_type() == "Constant")
                            {
                                auto input_map = (*ins->getGNode())["subgraph_input_map"];
                                NNFUSION_CHECK(input_map.is_valid());
                                inputs[ins->get_outputs()[0]->get_name()] = input_map.as<int>();
                            }
                        }
                    return inputs;
                }

                static std::vector<ir::Instruction::Pointer>
                    get_fused_kernel(const ir::Program& program)
                {
                    std::vector<ir::Instruction::Pointer> result;
                    for (auto blk : program)
                    {
                        for (auto ins : *blk)
                        {
                            auto kernel = ins->getKernel();
                            // neglect the Constant copy
                            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Result" ||
                                ins->getGNode()->get_op_type() == "Constant")
                                continue;
                            if (kernel->get_kernel_type() == "cuda")
                                result.push_back(ins);
                            else if (kernel->get_kernel_type() == "cuda_lib")
                            {
                                auto op = ins->getGNode()->get_op_ptr();
                                for (auto input : ins->get_inputs())
                                {
                                    std::cout << input->get_shape() << std::endl;
                                }
                                std::cout << ins->getGNode()->get_op_type() << "\n";
                                NNFUSION_CHECK_FAIL();
                            }
                        }
                    }
                    return result;
                }

                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
                {
                    auto type = tensor->get_element_type().c_type_string();
                    size_t offset = tensor->get_pool_offset();
                    return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) +
                           "+" + std::to_string(offset) + ")";
                }
            };
        }
    }
}
