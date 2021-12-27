#include "controlflow_emitter.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace kernels;

std::pair<cuda::dim3, cuda::dim3>
    cuda::ControlFlowEmitter::get_subgraph_launch_config(ir::BasicBlock::Pointer instructions)
{
    cuda::dim3 block_dim{1, 1, 1}, grid_dim{1, 1, 1};
    for (auto ins : *instructions)
    {
        auto cuda_kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        bool is_block = dynamic_pointer_cast<cuda::BlockCudaEmitter>(ins->getKernel()) != nullptr;
        auto dimb = cuda_kernel->get_block_dim(), dimg = cuda_kernel->get_grid_dim();
        if (!is_block && (dimb.y + dimb.z + dimg.y + dimg.z) != 4)
            NNFUSION_CHECK_FAIL() << ins->getGNode()->get_op_type();
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

std::string
    cuda::ControlFlowEmitter::get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
{
    auto type = tensor->get_element_type().c_type_string();
    NNFUSION_CHECK(tensor->get_pool_offset() != SIZE_MAX) << tensor->get_name();
    NNFUSION_CHECK(m_pool_offset.count(tensor->get_pool())) << tensor->get_pool();
    size_t offset = tensor->get_pool_offset() + m_pool_offset[tensor->get_pool()];
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
            else if (dynamic_pointer_cast<BlockCudaEmitter>(kernel) != nullptr)
            {
                auto ptr = dynamic_pointer_cast<BlockCudaEmitter>(kernel);
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

ir::BasicBlock::Pointer cuda::ControlFlowEmitter::create_param_map(
    const ir::Program& program, const std::unordered_map<std::string, int>& subgraph_output_map)
{
    ir::BasicBlock::Pointer result = std::make_shared<ir::BasicBlock>();
    ir::BasicBlock all_instructions;
    for (auto blk : program)
        for (auto ins : *blk)
            all_instructions.push_back(ins);

    auto input_map = get_subgraph_inputs(program);
    for (auto ins : all_instructions)
    {
        auto kernel = ins->getKernel();
        auto type = ins->getGNode()->get_op_type();
        if (type == "Constant" || type == "Parameter")
        {
            auto mapping = (*ins->getGNode())["subgraph_input_map"];
            NNFUSION_CHECK(mapping.is_valid());
            auto index = mapping.as<int>();
            if (index == -1)
                m_param_map[ins->get_outputs()[0]] = "&i"; // only defined for loop
            else
                m_param_map[ins->get_outputs()[0]] = "input" + std::to_string(index);
        }
        else if (kernel == nullptr || type == "Result")
            continue;
        else if (type == "Broadcast" &&
                 ins->get_inputs()[0]->size() == ins->get_outputs()[0]->size() &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)))
        {
            m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]];
        }
        else if (type == "Reshape" &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)) &&
                 !dynamic_pointer_cast<op::Reshape>(ins->getGNode()->get_op_ptr())
                      ->get_is_layout_change())
        {
            m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]];
        }
        else if (type == "GatherV2" &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)))
        {
            auto stride = ins->get_outputs()[0]->size(/*in_byte*/ false);
            m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]] + "+*(" +
                                                 m_param_map[ins->get_inputs()[1]] + ")*" +
                                                 std::to_string(stride);
        }
        else if (dynamic_pointer_cast<CudaEmitter>(kernel) == nullptr)
        {
            for (auto input : ins->get_inputs())
                std::cout << input->get_shape() << " ";
            std::cout << endl;
            for (auto input : ins->get_outputs())
                std::cout << input->get_shape() << " ";
            std::cout << endl;
            std::cout << ins->getGNode()->get_op_type() << "\n";
            NNFUSION_CHECK_FAIL();
        }
        else
        {
            for (const auto& tensor : ins->get_inputs())
            {
                if (m_param_map.count(tensor))
                    continue;
                else
                    m_param_map[tensor] = get_workspace_tensor(tensor);
            }
            for (const auto& tensor : ins->get_outputs())
            {
                if (m_param_map.count(tensor))
                    continue;
                else if (subgraph_output_map.count(tensor->get_name(false)))
                {
                    auto output_index = subgraph_output_map.at(tensor->get_name(false));
                    m_param_map[tensor] = "output" + std::to_string(output_index);
                }
                else
                    m_param_map[tensor] = get_workspace_tensor(tensor);
            }
            if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
                for (auto tensor : kernel->m_context->tensors)
                    m_param_map[tensor] = get_workspace_tensor(tensor);
            result->push_back(ins);
        }
    }
    return result;
}
