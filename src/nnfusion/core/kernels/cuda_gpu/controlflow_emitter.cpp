#include "controlflow_emitter.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

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

size_t cuda::ControlFlowEmitter::get_subgraph_shared_memory(const ir::Program& program)
{
    for (auto blk : program)
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            if (dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            {
                auto ptr = dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel);
                m_shared_memory_size = max(m_shared_memory_size, ptr->get_shared_memory_size());
            }
            else if (dynamic_pointer_cast<ControlFlowEmitter>(kernel) != nullptr)
            {
                auto ptr = dynamic_pointer_cast<ControlFlowEmitter>(kernel);
                m_shared_memory_size = max(m_shared_memory_size, ptr->m_shared_memory_size);
            }
        }
    return m_shared_memory_size;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_device_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_body"));
    auto& lu = *_lu;
    is_emitting_block_kernel = true;
    FunctionUnit_p fu = this->emit_source();
    is_emitting_block_kernel = false;
    lu << fu->body_unit->get_code() << "\n";
    return _lu;
}

void cuda::ControlFlowEmitter::allocate_shared_memory(LanguageUnit_p _lu)
{
    if (is_emitting_block_kernel)
        return;
    auto& lu = *_lu;
    if (m_shared_memory_size == 0)
    {
        lu << "char * shared_buffer = nullptr;\n";
    }
    else
    {
        lu << "__shared__ char shared_buffer[" + std::to_string(m_shared_memory_size) + "];\n";
    }
}