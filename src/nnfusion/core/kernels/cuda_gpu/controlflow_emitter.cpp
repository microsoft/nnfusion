#include "controlflow_emitter.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace kernels;
using namespace nnfusion::async;

DECLARE_bool(floop_in_c);
DEFINE_bool(ffast_barrier, false, "Use Rammer's fast barrier in control flow codegen");

LanguageUnit_p cuda::ControlFlowEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    set_launch_config();

    auto gnode = m_context->gnode;
    string stream_name = "0";
    if (gnode && (*gnode)["Async_info"].is_valid())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        if (async_info.execution_stream != nullptr)
            stream_name = async_info.execution_stream->get_name();
    }

    //set stream during codegen
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    if (FLAGS_ffast_barrier) {
        for (size_t i = 0; i < m_context->tensor_names.size(); i++) {
            if (m_context->tensors[i]->is_memset()) { // be_state_buffer
                names.push_back(m_context->tensor_names[i]);
            }
        }
        names.push_back("0"); // state_base
    }
    // names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());
    lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
       << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, " << stream_name
       << ">>>(" << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_function_signature()
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

    if (FLAGS_ffast_barrier) {
        params.push_back("volatile int32_t* be_state_buffer");
        params.push_back("int32_t state_base");
    }

    set_launch_config();
    emit_function_body();
    lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
       << ") __global__ void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_block_kernel_call(std::vector<std::string> params)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_call"));
    auto& lu = *_lu;
    if (FLAGS_ffast_barrier) {
        params.push_back("be_state_buffer");
        params.push_back("state_base");
    }
    params.push_back("shared_buffer");
    lu << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";
    return _lu;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_device_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_sig"));
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
    if (FLAGS_ffast_barrier) {
        params.push_back("volatile int32_t* be_state_buffer");
        params.push_back("int32_t& state_base");
    }
    params.push_back("char* shared_buffer");
    lu << "__device__ __noinline__ void " << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_block_executor_instruction_step_to(int max_block)
{
    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;
    lu << "state_base++;\n";
    lu << "if (blockIdx.x < " << max_block << ") " 
       << "BlockFusion_step_to_device_function(be_state_buffer, blockIdx.x, state_base);\n";

    return _lu;
}

LanguageUnit_p cuda::ControlFlowEmitter::emit_block_executor_instruction_wait_for(int max_block_last, int max_block_cur)
{
    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;
    if (max_block_cur != -1) {
        lu << "if (blockIdx.x < " << max_block_cur << ")";
        lu.block_begin();
    }
    lu << "if (threadIdx.x < " << max_block_last << ") while (be_state_buffer[threadIdx.x] < state_base);\n";
    lu << "__syncthreads();\n";
    if (max_block_cur != -1) {
        lu.block_end();
    }
    return _lu;
}

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
                NNFUSION_CHECK(input_map.is_valid()) << "invalid input: " << ins->getGNode()->get_name();
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

size_t cuda::ControlFlowEmitter::get_kernel_shared_memory(std::shared_ptr<KernelEmitter> kernel) {
    if (dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
    {
        auto ptr = dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel);
        return ptr->get_shared_memory_size();
    }
    else if (dynamic_pointer_cast<BlockCudaEmitter>(kernel) != nullptr)
    {
        auto ptr = dynamic_pointer_cast<BlockCudaEmitter>(kernel);
        return ptr->get_shared_memory_size();
    }
    else if (dynamic_pointer_cast<ControlFlowEmitter>(kernel) != nullptr)
    {
        auto ptr = dynamic_pointer_cast<ControlFlowEmitter>(kernel);
        return ptr->m_shared_memory_size;
    }
    return 0;
}

size_t cuda::ControlFlowEmitter::get_subgraph_shared_memory(const ir::Program& program)
{
    for (auto blk : program)
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            m_shared_memory_size = max(m_shared_memory_size, get_kernel_shared_memory(kernel));
        }
    return m_shared_memory_size;
}

size_t cuda::ControlFlowEmitter::get_inst_max_shared_memory(nnfusion::ir::BasicBlock::Pointer bb, int start_id, int end_id) {
    if (end_id == -1) end_id = bb->size();
    size_t mx = 0;
    for (int i = start_id; i < end_id; i++)
        mx = max(mx, get_kernel_shared_memory(bb->at(i)->getKernel()));
    return mx;
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

void cuda::ControlFlowEmitter::allocate_shared_memory(LanguageUnit_p _lu, size_t shared_memory_size)
{
    if (is_emitting_block_kernel)
        return;
    auto& lu = *_lu;
    if (shared_memory_size == 0)
    {
        lu << "char * shared_buffer = nullptr;\n";
    }
    else
    {
        lu << "__shared__ char shared_buffer[" + std::to_string(shared_memory_size) + "];\n";
    }
}

void cuda::ControlFlowEmitter::allocate_shared_memory(LanguageUnit_p _lu) {
    allocate_shared_memory(_lu, m_shared_memory_size);
}

ir::BasicBlock::Pointer cuda::ControlFlowEmitter::create_param_map(
    const ir::Program& program, const std::unordered_map<std::string, int>& subgraph_output_map, bool call_on_cuda)
{
    ir::BasicBlock::Pointer result = std::make_shared<ir::BasicBlock>();
    ir::BasicBlock all_instructions;
    for (auto blk : program)
        for (auto ins : *blk)
            all_instructions.push_back(ins);

    auto input_map = get_subgraph_inputs(program);
    std::map<nnfusion::descriptor::Tensor::Pointer, std::string> scalar_map;
    
    for (auto ins : all_instructions)
    {
        auto kernel = ins->getKernel();
        auto type = ins->getGNode()->get_op_type();
        if (type == "Constant" || type == "Parameter")
        {
            auto mapping = (*ins->getGNode())["subgraph_input_map"];
            NNFUSION_CHECK(mapping.is_valid());
            auto index = mapping.as<int>();
            if (index == -1) {
                if (FLAGS_floop_in_c) {
                    m_param_map[ins->get_outputs()[0]] = "i_dev"; // only defined for loop
                    scalar_map[ins->get_outputs()[0]] = "i";
                } else {
                    m_param_map[ins->get_outputs()[0]] = "&i"; // only defined for loop
                    scalar_map[ins->get_outputs()[0]] = "i";
                }
                (*ins->getGNode())["output_unused"] = std::vector<int>();
            }
            else {
                m_param_map[ins->get_outputs()[0]] = "input" + std::to_string(index);
                if (type == "Constant" && shape_size(ins->get_outputs()[0]->get_shape()) == 1) {
                    auto const_op = dynamic_pointer_cast<op::Constant>(ins->getGNode()->get_op_ptr());
                    scalar_map[ins->get_outputs()[0]] = const_op->get_value_strings()[0];
                    (*ins->getGNode())["output_unused"] = std::vector<int>();
                }
            }
            continue;
        }
        else if (kernel == nullptr || type == "Result")
            continue;
        else if (type == "Broadcast" &&
                 ins->get_inputs()[0]->size() == ins->get_outputs()[0]->size() &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)))
        {
            m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]];
            continue;
        }
        else if (type == "Reshape" &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)) &&
                 !dynamic_pointer_cast<op::Reshape>(ins->getGNode()->get_op_ptr())
                      ->get_is_layout_change())
        {
            m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]];
            continue;
        }
        else if (type == "GatherV2" &&
                 !subgraph_output_map.count(ins->get_outputs()[0]->get_name(false)))
        {
            auto index_edge = ins->getGNode()->get_in_edge(1);
            auto index_gnode = index_edge->get_src();
            auto stride = ins->get_outputs()[0]->size(/*in_byte*/ false);
            if (scalar_map.find(index_gnode->get_output_tensor_ptr(0)) != scalar_map.end()) {
                m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]] + "+(" + scalar_map[index_gnode->get_output_tensor_ptr(0)] + ")*" + std::to_string(stride);
                (*index_gnode)["output_unused"].as<std::vector<int>>().push_back(index_edge->get_src_output());
                continue;
            } else if (call_on_cuda) {
                m_param_map[ins->get_outputs()[0]] = m_param_map[ins->get_inputs()[0]] + "+*(" +
                                                    m_param_map[ins->get_inputs()[1]] + ")*" +
                                                    std::to_string(stride);
                continue;
            }
        } 
        else if (type == "ElementWiseFused") {
            auto fused_gnode = std::static_pointer_cast<graph::FusedGNode>(ins->getGNode());
            auto ctxs = fused_gnode->get_op_contexts();
            bool all_input_scalar = true;
            for (auto c: ctxs) {
                std::vector<std::string> input_names;
                for (size_t i = 0; i < c->inputs.size(); i++) {
                    if (scalar_map.find(c->inputs[i]) == scalar_map.end()) {
                        all_input_scalar = false;
                        break;
                    }
                    input_names.push_back(scalar_map[c->inputs[i]]);
                }
                if (!all_input_scalar) break;
                std::string ele_op_type = c->op->get_op_type();
                // WARNING: be careful, code is written by copilot
                if (ele_op_type == "Multiply") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 2);
                    scalar_map[c->outputs[0]] = "(" + input_names[0] + ")*(" + input_names[1] + ")";
                } else if (ele_op_type == "Add") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 2);
                    scalar_map[c->outputs[0]] = "(" + input_names[0] + ")+(" + input_names[1] + ")";
                } else if (ele_op_type == "Subtract") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 2);
                    scalar_map[c->outputs[0]] = "(" + input_names[0] + ")-(" + input_names[1] + ")";
                } else if (ele_op_type == "Divide") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 2);
                    scalar_map[c->outputs[0]] = "(" + input_names[0] + ")/(" + input_names[1] + ")";
                } else if (ele_op_type == "Negative") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 1);
                    scalar_map[c->outputs[0]] = "-(" + input_names[0] + ")";
                } else if (ele_op_type == "Exp") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 1);
                    scalar_map[c->outputs[0]] = "exp(" + input_names[0] + ")";
                } else if (ele_op_type == "Log") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 1);
                    scalar_map[c->outputs[0]] = "log(" + input_names[0] + ")";
                } else if (ele_op_type == "Sqrt") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 1);
                    scalar_map[c->outputs[0]] = "sqrt(" + input_names[0] + ")";
                } else if (ele_op_type == "Sigmoid") {
                    NNFUSION_CHECK(c->outputs.size() == 1);
                    NNFUSION_CHECK(input_names.size() == 1);
                    NNFUSION_CHECK_FAIL() << "unsupported: " << ele_op_type;
                } else {
                    NNFUSION_CHECK_FAIL() << "unsupported: " << ele_op_type;
                }
            }
            bool all_output_scalar = true;
            for (auto c: ctxs) {
                for (auto outp: c->outputs) {
                    if (scalar_map.find(outp) == scalar_map.end()) {
                        all_output_scalar = false;
                        break;
                    }
                }
                if (!all_output_scalar) break;
            }
            if (all_output_scalar) {
                (*ins->getGNode())["output_unused"] = std::vector<int>();
            }
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
        for (const auto& tensor : ins->get_inputs())
        {
            if (m_param_map.count(tensor))
                continue;
            else
                m_param_map[tensor] = get_workspace_tensor(tensor);
        }
        if (ins->getKernel()->m_context->annotations != nullptr)
        {
            for (auto pair : ins->getKernel()->m_context->annotations->get_in_place_oi_pairs())
            {
                auto output = ins->get_outputs()[pair.output];
                auto input = ins->get_inputs()[pair.input];
                if (pair.force_inplace)
                {
                    m_param_map[output] = m_param_map[input];
                }
            }
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
    // remove unused kernels
     ir::BasicBlock::Pointer result_used = std::make_shared<ir::BasicBlock>();
    for (auto ins: *result) {
        bool should_remove = false;
        bool all_output_scalar = true;
        for (auto outp: ins->get_outputs()) {
            if (scalar_map.find(outp) == scalar_map.end()) {
                all_output_scalar = false;
                break;
            }
        }
        if (all_output_scalar) {
            if ((*ins->getGNode())["output_unused"].as<std::vector<int>>().size() ==  ins->getGNode()->get_out_edges().size()) {
                NNFUSION_LOG(INFO) << "unused instruction: " << ins->getKernel()->m_context->gnode->get_name();
                should_remove = true;
            }
        }
        if (!should_remove) {
            result_used->push_back(ins);
        }
    }

    return result_used;
}
