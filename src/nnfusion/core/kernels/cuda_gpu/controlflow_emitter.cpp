#include "controlflow_emitter.hpp"

using namespace nnfusion;
using namespace kernels;

std::pair<cuda::dim3, cuda::dim3>
    cuda::ControlFlowEmitter::get_subgraph_launch_config(const ir::Program& program)
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

std::map<std::string, int> cuda::ControlFlowEmitter::get_subgraph_inputs(const ir::Program& program)
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

std::vector<ir::Instruction::Pointer>
    cuda::ControlFlowEmitter::get_fused_kernel(const ir::Program& program)
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

std::string
    cuda::ControlFlowEmitter::get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
{
    auto type = tensor->get_element_type().c_type_string();
    size_t offset = tensor->get_pool_offset();
    if (tensor->get_name(false) == "recursion_stack")
    {
        return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) +
               "+ $stack_size$)";
    }
    else
    {
        return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) + "+" +
               std::to_string(offset) + ")";
    }
}

std::string cuda::ControlFlowEmitter::get_launch_bound(nnfusion::ir::Instruction::Pointer ins)
{
    auto type = ins->getGNode()->get_op_type();
    if (type == "If" || type == "Loop" || type == "Recursion" || type == "FuncForward")
        return "";
    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
    cuda::dim3 grid = kernel->get_grid_dim();
    return "if (blockIdx.x < " + std::to_string(grid.x) + " && blockIdx.y < " +
           std::to_string(grid.y) + ")\n";
}
