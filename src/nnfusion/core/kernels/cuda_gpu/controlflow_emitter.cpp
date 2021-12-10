#include "controlflow_emitter.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace kernels;

std::pair<cuda::dim3, cuda::dim3>
    cuda::ControlFlowEmitter::get_subgraph_launch_config(const ir::Program& program)
{
    cuda::dim3 block_dim{1, 1, 1}, grid_dim{1, 1, 1};
    auto instructions = get_fused_kernel(program);
    for (auto ins : instructions)
    {
        auto cuda_kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        bool is_block = dynamic_pointer_cast<cuda::BlockCudaEmitter>(ins->getKernel()) != nullptr;
        auto dimb = cuda_kernel->get_block_dim(), dimg = cuda_kernel->get_grid_dim();
        // if (!is_block && (dimb.y + dimb.z + dimg.y + dimg.z) != 4)
        //     NNFUSION_CHECK_FAIL() << ins->getGNode()->get_op_type();
        block_dim.x = max(block_dim.x, dimb.x * dimb.y * dimb.z);
        grid_dim.x = max(grid_dim.x, dimg.x * dimg.y * dimg.z);
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
            auto type = ins->getGNode()->get_op_type();
            if ((type == "Reshape" || type == "Broadcast") &&
                ins->get_inputs()[0]->size() == ins->get_outputs()[0]->size())
                continue;
            // neglect the Constant copy
            if (kernel == nullptr || type == "Result" || type == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
                result.push_back(ins);
            else if (kernel->get_kernel_type() == "cuda_lib")
            {
                for (auto input : ins->get_inputs())
                    std::cout << input->get_shape() << " ";
                std::cout << endl;
                for (auto input : ins->get_outputs())
                    std::cout << input->get_shape() << " ";
                std::cout << endl;
                std::cout << ins->getGNode()->get_op_type() << "\n";
                // NNFUSION_CHECK_FAIL();
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
    return "if (blockIdx.x < " + std::to_string(grid.x * grid.y * grid.z) + ")\n";
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

void cuda::ControlFlowEmitter::bypass_instructions(const ir::Program& program)
{
    std::vector<nnfusion::ir::Instruction::Pointer> instructions;
    for (auto blk : program)
        for (auto ins : *blk)
            instructions.push_back(ins);

    std::unordered_map<nnfusion::descriptor::Tensor::Pointer, nnfusion::descriptor::Tensor::Pointer>
        replace;
    for (auto ins : instructions)
    {
        auto kernel = ins->getKernel();
        auto type = ins->getGNode()->get_op_type();
        if ((type == "Reshape" || type == "Broadcast") &&
            ins->get_inputs()[0]->size() == ins->get_outputs()[0]->size())
            replace[ins->get_outputs()[0]] = ins->get_inputs()[0];
        else
        {
            for (auto& input : ins->get_inputs())
                if (replace.count(input))
                    input = replace[input];
        }
    }
}
