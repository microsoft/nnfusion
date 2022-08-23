// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "if.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DEFINE_bool(fif_launch_then_else, false, "launch kernels in both then branch and else branch");
DEFINE_int32(ffused_max_grid, 512, "max griddim in fused kernels");

cuda::If::If(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_IfOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::If>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_then_branch_tu = op->get_then_branch_tu();
    m_else_branch_tu = op->get_else_branch_tu();
    size_t size0 = 0, size1 = 0;
    std::unordered_map<std::string, size_t> then_branch_pool_offset, else_branch_pool_offset;
    for (auto& pair : m_then_branch_tu->memory_allocator_factory->get_allocator_list())
    {
        then_branch_pool_offset[pair.second->get_name()] = size0;
        size0 += pair.second->max_allocated();
    }
    for (auto& pair : m_else_branch_tu->memory_allocator_factory->get_allocator_list())
    {
        else_branch_pool_offset[pair.second->get_name()] = size1;
        size1 += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape({std::max(size0, size1)}), nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_shared_memory_size = max(get_subgraph_shared_memory(m_then_branch_tu->program),
                               get_subgraph_shared_memory(m_else_branch_tu->program));
    m_pool_offset = then_branch_pool_offset;
    m_then_branch_instructions = create_param_map(m_then_branch_tu->program, op->get_output_map());
    m_pool_offset = else_branch_pool_offset;
    m_else_branch_instructions = create_param_map(m_else_branch_tu->program, op->get_output_map());
    // Hardcoded fusion rule: fuse all kernels with shm_size=0
    m_then_kernel_groups.clear();
    for (int i = 0; i < m_then_branch_instructions->size(); i++) {
        size_t shm_size = get_kernel_shared_memory((*m_then_branch_instructions).at(i)->getKernel());
        if (shm_size > 0) {
            if (m_then_kernel_groups[m_then_kernel_groups.size() - 1].size() == 0) {
                m_then_kernel_groups[m_then_kernel_groups.size() - 1].push_back(i);
            } else {
                m_then_kernel_groups.push_back(std::vector<int>({i}));
            }
            m_then_kernel_groups.push_back(std::vector<int>());
        } else {
            if (i == 0) {
                m_then_kernel_groups.push_back(std::vector<int>());
            }
            m_then_kernel_groups[m_then_kernel_groups.size() - 1].push_back(i);
        }
    }
    if (m_then_kernel_groups[m_then_kernel_groups.size() - 1].size() == 0)
        m_then_kernel_groups.pop_back();
    NNFUSION_LOG(INFO) << "then groups from size " << m_then_branch_instructions->size();
    for (auto group: m_then_kernel_groups) {
        for (auto x: group) printf("%d ", x);
        printf("\n");
    }
    m_else_kernel_groups.clear();
    for (int i = 0; i < m_else_branch_instructions->size(); i++) {
        size_t shm_size = get_kernel_shared_memory((*m_else_branch_instructions).at(i)->getKernel());
        if (shm_size > 0) {
            if (m_else_kernel_groups[m_else_kernel_groups.size() - 1].size() == 0) {
                m_else_kernel_groups[m_else_kernel_groups.size() - 1].push_back(i);
            } else {
                m_else_kernel_groups.push_back(std::vector<int>({i}));
            }
            m_else_kernel_groups.push_back(std::vector<int>());
        } else {
            if (i == 0) {
                m_else_kernel_groups.push_back(std::vector<int>());
            }
            m_else_kernel_groups[m_else_kernel_groups.size() - 1].push_back(i);
        }
    }
    if (m_else_kernel_groups[m_else_kernel_groups.size() - 1].size() == 0)
        m_else_kernel_groups.pop_back();
    NNFUSION_LOG(INFO) << "else groups";
    for (auto group: m_else_kernel_groups) {
        for (auto x: group) printf("%d ", x);
        printf("\n");
    }
    // NNFUSION_CHECK_FAIL();
}

void cuda::If::generate_branch_fused_kernel(LanguageUnit_p _lu, ir::BasicBlock::Pointer instructions, int start_id, int end_id)
{
    auto& lu = *_lu;
    if (end_id == -1) end_id = instructions->size();
    for (int i = start_id; i < end_id; i++)
    {
        auto ins = instructions->at(i);
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
            params.push_back(m_param_map[tensor]);
        for (auto tensor : ins->get_outputs())
            params.push_back(m_param_map[tensor]);
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(m_param_map[tensor]);
        std::string kernel_call = kernel->emit_block_kernel_call(params)->get_code();
        int griddim = kernel->get_grid_dim().x;
        if (griddim > FLAGS_ffused_max_grid) {
            size_t start_pos = kernel_call.find("blockIdx.x");
            kernel_call = replace_one(kernel_call, "blockIdx.x", "blockIdx.x + start_block");
            std::string launch_bound = get_launch_bound(ins);
            launch_bound = replace_one(launch_bound, "blockIdx.x", "blockIdx.x + start_block");
            lu << "for (int start_block = 0; start_block < " << griddim << "; start_block += gridDim.x)\n";
            lu.block_begin();
            lu << launch_bound;
            lu.block_begin();
            lu << kernel_call;
            lu.block_end();
            lu.block_end();
        } else {
            lu << get_launch_bound(ins);
            lu.block_begin();
            lu << kernel_call;
            lu.block_end();
        }
        if (i != end_id - 1)
            lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::If::generate_branch_seperate_function(const std::string& outer_control, std::shared_ptr<descriptor::Tensor> cond_tensor, std::shared_ptr<ir::Instruction> ins) {
    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
    std::string func_name = kernel->get_function_name() + "_branch_wrapper";
    LanguageUnit_p _lu(new LanguageUnit(func_name));
    auto& lu = *_lu;
    // function signature
    std::vector<std::string> params;
    std::vector<std::string> param_names;
    auto inputs = ins->get_inputs();
    inputs.insert(inputs.begin(), cond_tensor);
    for (size_t i = 0; i < inputs.size(); i++)
    {
        stringstream ss;
        ss << inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }
    for (size_t i = 1; i < inputs.size(); i++) // skip the cond tensor
    {
        stringstream ss;
        ss << "input" << i;
        param_names.push_back(ss.str());
    }
    auto& outputs = ins->get_outputs();
    for (size_t i = 0; i < outputs.size(); i++)
    {
        stringstream ss;
        ss << outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        stringstream ss;
        ss << "output" << i;
        param_names.push_back(ss.str());
    }

    lu << "__global__ void " << func_name << "(" << join(params, ", ") << ")";
    lu.block_begin();
    lu << outer_control;
    lu.block_begin();
    allocate_shared_memory(_lu, get_kernel_shared_memory(kernel));
    LanguageUnit_p block_kernel_call = kernel->emit_block_kernel_call(param_names);
    lu << block_kernel_call->get_code();
    lu.block_end();
    lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::If::generate_branch_fused_function(const std::string& outer_control, bool else_branch, int start_id, int end_id) {
    std::string func_name = get_function_name() + "_branch_wrapper_" + (else_branch ? "else_" : "then_") + std::to_string(start_id) + "_to_" + std::to_string(end_id - 1);
    LanguageUnit_p _lu(new LanguageUnit(func_name));
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

    // set_launch_config();
    auto instructions = else_branch ? m_else_branch_instructions : m_then_branch_instructions;
    emit_function_body();
    lu << "__global__ void " << func_name << "(" << join(params, ", ") << ")";
    lu.block_begin();
    lu << outer_control;
    {
        size_t shm_size = get_inst_max_shared_memory(instructions, start_id, end_id);
        lu.block_begin();
        allocate_shared_memory(_lu, shm_size);
        generate_branch_fused_kernel(_lu, instructions, start_id, end_id);
        lu.block_end();
    }
    lu.block_end();
    return _lu;
}

void cuda::If::emit_kernel_wrapper(std::shared_ptr<ir::Instruction> ins, LanguageUnit &lu) {
    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
    std::string wrapper_func_name = kernel->get_function_name() + "_branch_wrapper";
    cuda::dim3 grid_dim = kernel->get_grid_dim();
    cuda::dim3 block_dim = kernel->get_block_dim();
    std::vector<string> params;
    params.push_back("input0");
    for (auto tensor : ins->get_inputs())
        params.push_back(m_param_map[tensor]);
    for (auto tensor : ins->get_outputs())
        params.push_back(m_param_map[tensor]);
    lu << wrapper_func_name;
    lu << "<<<dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), dim3("
    << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")>>>(" << join(params, ", ") << ");\n";
}

void cuda::If::emit_branch_wrapper(bool else_branch, int start_id, int end_id, LanguageUnit &lu, bool emit_all_args) {
    std::string func_name = get_function_name() + "_branch_wrapper_" + (else_branch ? "else_" : "then_") + std::to_string(start_id) + "_to_" + std::to_string(end_id - 1);
    auto instructions = else_branch ? m_else_branch_instructions : m_then_branch_instructions;
    int grid_x = 1, block_x = 1;
    for (int i = start_id; i < end_id; i++) {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(instructions->at(i)->getKernel());
        grid_x = max(grid_x, kernel->get_grid_dim().x);
        block_x = max(block_x, kernel->get_block_dim().x);
    }
    if (start_id + 1 < end_id) // fused kernel with cooperative group
    {
        grid_x = min(grid_x, FLAGS_ffused_max_grid);
    }
    cuda::dim3 grid_dim = dim3(grid_x, 1, 1);
    cuda::dim3 block_dim = dim3(block_x, 1, 1);
    if (emit_all_args) {
        std::vector<string> params;
        for (size_t i = 0; i < m_context->inputs.size(); i++)
        {
            stringstream ss;
            ss << "&input" << i;
            params.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "&output" << i;
            params.push_back(ss.str());
        }
        lu << "void* args[] = {" << join(params, ", ") << "};\n";
    }
    lu << "CUDA_SAFE_CALL(cudaLaunchCooperativeKernel(" 
       << "(const void*) " << func_name << ", "
       << "dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), "
       << "dim3(" << block_dim.x << "," << block_dim.y << ", " << block_dim.z << "), "
       << "args, (size_t) 0, (cudaStream_t)0));\n";
}

LanguageUnit_p cuda::If::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    if (FLAGS_fif_launch_then_else) {
        auto& lu = *_lu;
        bool emit_all_args = true;
        lu << "// then branch\n";
        for (auto group: m_then_kernel_groups) {
            if (group.size() == 1) {
                auto ins = m_then_branch_instructions->at(group[0]);
                emit_kernel_wrapper(ins, lu);
            } else {
                emit_branch_wrapper(false, group[0], group[group.size() - 1] + 1, lu, emit_all_args);
                emit_all_args = false;
            }
        }
        lu << "// else branch\n";
        for (auto group: m_else_kernel_groups) {
            if (group.size() == 1) {
                auto ins = m_else_branch_instructions->at(group[0]);
                emit_kernel_wrapper(ins, lu);
            } else {
                emit_branch_wrapper(true, group[0], group[group.size() - 1] + 1, lu, emit_all_args);
                emit_all_args = false;
            }
        }
    } else {
        auto& lu = *_lu;
        allocate_shared_memory(_lu);
        lu << "if (*input0) ";
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_then_branch_instructions);
        lu.block_end();
        lu << "else ";
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_else_branch_instructions);
        lu.block_end();
    }
    return _lu;
}

LanguageUnit_p cuda::If::emit_function_call()
{
    if (!FLAGS_fif_launch_then_else) {
        return CudaEmitter::emit_function_call();
    } else {
        return KernelEmitter::emit_function_call();
    }
}

void cuda::If::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_then_branch_instructions);
    auto cfg1 = get_subgraph_launch_config(m_else_branch_instructions);
    m_blockDim = maxdim3(cfg0.first, cfg1.first);
    m_gridDim = maxdim3(cfg0.second, cfg1.second);
}

LanguageUnit_p cuda::If::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    if (FLAGS_fif_launch_then_else) {
        for (auto group: m_then_kernel_groups) {
            if (group.size() == 1) {
                auto ins = m_then_branch_instructions->at(group[0]);
                auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                auto body = kernel->get_or_emit_source();
                auto block_kernel = kernel->emit_block_kernel();
                block_kernel->require(body->dep_unit);
                _lu->require(block_kernel);
                auto ins_kernel = generate_branch_seperate_function("if (*input0) ", m_context->inputs[0], ins);
                ins_kernel->require(block_kernel);
                _lu->require(ins_kernel);        
            } else {
                auto group_kernel = generate_branch_fused_function("if (*input0)", false, group[0], group[group.size() - 1] + 1);
                for (auto inst_id: group) {
                    auto ins = m_then_branch_instructions->at(inst_id);
                    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                    auto body = kernel->get_or_emit_source();
                    auto block_kernel = kernel->emit_block_kernel();
                    block_kernel->require(body->dep_unit);
                    group_kernel->require(block_kernel);
                }
                _lu->require(group_kernel);
            }
        }
        for (auto group: m_else_kernel_groups) {
            if (group.size() == 1) {
                auto ins = m_else_branch_instructions->at(group[0]);
                auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                auto body = kernel->get_or_emit_source();
                auto block_kernel = kernel->emit_block_kernel();
                block_kernel->require(body->dep_unit);
                _lu->require(block_kernel);
                auto ins_kernel = generate_branch_seperate_function("if (!(*input0)) ", m_context->inputs[0], ins);
                ins_kernel->require(block_kernel);
                _lu->require(ins_kernel);        
            } else {
                auto group_kernel = generate_branch_fused_function("if (!(*input0))", true, group[0], group[group.size() - 1] + 1);
                _lu->require(group_kernel);
                for (auto inst_id: group) {
                    auto ins = m_else_branch_instructions->at(inst_id);
                    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                    auto body = kernel->get_or_emit_source();
                    auto block_kernel = kernel->emit_block_kernel();
                    block_kernel->require(body->dep_unit);
                    _lu->require(block_kernel);
                }
                _lu->require(group_kernel);
            }
        }
    } else {
        for (auto ins : *m_then_branch_instructions)
        {
            auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
            auto body = kernel->get_or_emit_source();
            auto block_kernel = kernel->emit_block_kernel();
            block_kernel->require(body->dep_unit);
            _lu->require(block_kernel);
        }
        for (auto ins : *m_else_branch_instructions)
        {
            auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
            auto body = kernel->get_or_emit_source();
            auto block_kernel = kernel->emit_block_kernel();
            block_kernel->require(body->dep_unit);
            _lu->require(block_kernel);
        }
    }
    return _lu;
}

LanguageUnit_p cuda::If::emit_function_signature()
{
    if (!FLAGS_fif_launch_then_else) return ControlFlowEmitter::emit_function_signature();
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

    // the temp tensor have been included in input tensors. Duplicate here to align with KernelEmiter::emit_function_call();
    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        ss << "tensor" << i;
        params.push_back(ss.str());
    }

    set_launch_config();
    emit_function_body();
    lu << "void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}


REGISTER_KERNEL_EMITTER("If",                                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::If)
