// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recursion.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"
#include <regex>

using namespace nnfusion;
using namespace nnfusion::kernels;

DEFINE_bool(frecursive_stack, false, "recursive with manual stack");
DEFINE_bool(fstack_in_glb, false, "stack in global memory");
DEFINE_bool(fparallel_recursion, false, "parallel recursion");
DEFINE_int32(fparallel_recursion_min, -1, "parallel recursion min");
DEFINE_int32(fparallel_recursion_grid, -1, "parallel recursion grid");
DEFINE_int32(frecursive_max_depth, 20, "manual stack size");
DECLARE_bool(ffast_barrier);

cuda::FuncForward::FuncForward(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_FuncForwardOP";
    custom_tag = tag.str();
    m_workspace = allocate_tensor(Shape{0}, nnfusion::element::character);
    m_workspace->set_name("recursion_stack");
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
}

LanguageUnit_p cuda::FuncForward::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    return _lu;
}

LanguageUnit_p cuda::FuncForward::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    return _lu;
}

void cuda::FuncForward::set_launch_config()
{
}

LanguageUnit_p cuda::FuncForward::emit_block_kernel_call(std::vector<std::string> params)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_call"));
    auto& lu = *_lu;
    if (m_block_func_name == "")
        m_block_func_name = m_kernel_name + "_recursion";
    if (FLAGS_ffast_barrier) {
        params.push_back("be_state_buffer");
        params.push_back("state_base");
    }
    params.push_back("shared_buffer");

    lu << m_block_func_name + "_block_kernel(" << join(params, ", ") << ");\n";
    return _lu;
}

std::string cuda::FuncForward::m_block_func_name = "";

void cuda::FuncForward::update_context_from_gnode(std::shared_ptr<nnfusion::graph::GNode> gnode) {
    auto ctx = std::make_shared<KernelContext>(gnode);
    m_context = ctx;
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
}

cuda::Recursion::Recursion(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_RecursionOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::Recursion>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_loop_body_tu = op->get_body_tu();
    m_workspace_size = 0;
    for (auto& pair : m_loop_body_tu->memory_allocator_factory->get_allocator_list())
    {
        m_pool_offset[pair.second->get_name()] = m_workspace_size;
        m_workspace_size += pair.second->max_allocated();
    }
    size_t context_param_offset = 233333333;
    if (FLAGS_fstack_in_glb) {
        m_param_offset["s_label"] = m_workspace_size;
        m_workspace_size += sizeof(int) * FLAGS_frecursive_max_depth;
        for (auto input: m_context->inputs) {
            m_param_offset[input->get_name()] = m_workspace_size;
            m_workspace_size += sizeof(void*) * FLAGS_frecursive_max_depth;
        }
        for (auto output: m_context->outputs) {
            m_param_offset[output->get_name()] = m_workspace_size;
            m_workspace_size += sizeof(void*) * FLAGS_frecursive_max_depth;
        }
        context_param_offset = m_workspace_size;
        m_workspace_size += sizeof(void*) * FLAGS_frecursive_max_depth; // reserved for m_workspace tensor
    }
    if (FLAGS_fparallel_recursion) {
        m_workspace = allocate_tensor(Shape{m_workspace_size * FLAGS_frecursive_max_depth * (FLAGS_fparallel_recursion_grid / FLAGS_fparallel_recursion_min)}, nnfusion::element::character);
    } else {
        m_workspace = allocate_tensor(Shape{m_workspace_size * FLAGS_frecursive_max_depth}, nnfusion::element::character);
    }
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_param_offset[m_workspace->get_name()] = context_param_offset;
    m_block_func_name = move(FuncForward::m_block_func_name);
    NNFUSION_CHECK(!m_block_func_name.empty());
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    m_body_instructions = create_param_map(m_loop_body_tu->program, op->get_output_map(), true);
    if (FLAGS_ffast_barrier) {
        auto launch_config = get_subgraph_launch_config(m_body_instructions);
        dim3 grid_dim = launch_config.second;
        m_sync_tensor = std::make_shared<nnfusion::descriptor::Tensor>(
                    nnfusion::element::i32,
                    nnfusion::PartialShape(
                        {(size_t)grid_dim.x * (size_t)grid_dim.y * (size_t)grid_dim.z}),
                        get_function_name() + "_be_state_buffer",
                    nnfusion::NNFusion_DeviceType::CUDA_GPU);
        m_sync_tensor->set_memset(true, 0);
        m_sync_tensor->set_persistent(true);
        m_context->tensors.push_back(m_sync_tensor);
        m_context->tensor_names.push_back(m_sync_tensor->get_name());
    }
}

std::string cuda::Recursion::inline_kernel(std::shared_ptr<ir::Instruction> ins) {
    auto cuda_kernel = dynamic_pointer_cast<CudaEmitter>(ins->getKernel());
    std::string code = cuda_kernel->emit_device_function_body()->get_code();
    code = replace_all(code, "input", "IN_PUT_");
    code = replace_all(code, "output", "OUT_PUT_");
    auto inputs = ins->get_inputs();
    for (int i = 0; i < inputs.size(); i++) {
        stringstream ss_from;
        ss_from << "IN_PUT_" << i;
        std::string from = ss_from.str();
        NNFUSION_LOG(INFO) << "replace " << from << " to " << m_param_map[inputs[i]];
        code = replace_all(code, from, "(" + m_param_map[inputs[i]] + ")");
    }
    auto outputs = ins->get_outputs();
    for (int i = 0; i < outputs.size(); i++) {
        stringstream ss_from;
        ss_from << "OUT_PUT_" << i;
        std::string from = ss_from.str();
        NNFUSION_LOG(INFO) << "replace " << from << " to " << m_param_map[outputs[i]];
        code = replace_all(code, from, "(" + m_param_map[outputs[i]] + ")");
    }
    return code;
}

std::string to_local_block(std::string code) {
    code = "int exec_block_id;" + code; 
    std::string pattern = "if \\(blockIdx.x < [[:digit:]]+\\)";
    while (true) {
        std::smatch m;
        std::regex_search(code, m, std::regex(pattern));
        if (m.empty()) break;
        NNFUSION_LOG(INFO) << m.size() << "matches: " << m[0];;
        int max_block_size = std::stoi(m[0].str().substr(17, m[0].str().size() - 18));
        std::cout << "max_block_size: " << max_block_size << std::endl;
        std::cout << "----------------\n";
        if (max_block_size >= FLAGS_fparallel_recursion_min) {
            code = replace_all(code, m[0].str(), "for (int exec_block_id = local_block_id; exec_block_id < " + std::to_string(max_block_size) + "; exec_block_id += local_block_size)");
        } else {
            code = replace_all(code, m[0].str(), "exec_block_id = local_block_id; if (exec_block_id < " + std::to_string(max_block_size) + ")");
        }
    }
    code = replace_all(code, "blockIdx.x", "exec_block_id");
    code = replace_all(code, "Barrier()", "block_Barrier(local_block_id, local_block_size)");
    return code;
}

std::string cuda::Recursion::to_stack(std::string code, std::shared_ptr<ir::Instruction> ins) {
    std::vector<std::string> call_sites;
    std::regex pattern(FuncForward::m_block_func_name + "_block_kernel.*?;");
    // https://stackoverflow.com/questions/21667295/how-to-match-multiple-results-using-stdregex
    string::const_iterator search_start(code.cbegin());
    smatch match;
    while (std::regex_search(search_start, code.cend(), match, pattern)) {
        call_sites.push_back(match[0]);
        search_start = match.suffix().first;
    }
    if (call_sites.size() == 0) return code;
    if (FLAGS_fparallel_recursion) {
        code = to_local_block(code);
    }
    std::vector<std::vector<std::string>> caller_params;
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    for (auto call: call_sites) {
        std::vector<std::string> params;
        call = call.substr((FuncForward::m_block_func_name + "_block_kernel(").length());
        std::cout << "params: " << call << ": ";
        size_t pos = 0;
        while ((pos = call.find(", ")) != std::string::npos) {
            std::string token = call.substr(0, pos);
            params.push_back(token);
            call.erase(0, pos + 2);
        }
        params.push_back(call.substr(0, call.length() - 2));
        for (auto param: params)
            std::cout << param << "|";
        std::cout << "\n";
        caller_params.push_back(params);
    }
    std::vector<int> input_need_stack;
    for (int i = 0; i < m_context->inputs.size(); i++) {
        bool all_same = true;
        std::string std = "(input" + std::to_string(i) + ")";
        for (int j = 0; j < caller_params.size(); j++) {
            if (caller_params[j][i] != std) all_same = false;
        }
        if (!all_same) {
            input_need_stack.push_back(i);
        }
    }
    std::vector<int> output_need_stack;
    for (int i = 0; i < m_context->outputs.size(); i++) {
        bool all_same = true;
        std::string std = "(output" + std::to_string(i) + ")";
        for (int j = 0; j < caller_params.size(); j++) {
            if (caller_params[j][i] != std) all_same = false;
        }
        if (!all_same) {
            output_need_stack.push_back(i);
        }
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_to_stack"));
    LanguageUnit& lu = *_lu;
    auto inputs = m_context->inputs;
    auto outputs = m_context->outputs;
    if (FLAGS_fstack_in_glb) {
        lu << "volatile int* s_label = (volatile int*) input" + std::to_string(m_context->inputs.size() - 1) << " + " << m_param_offset["s_label"] << ";\n";

        for (auto input_id: input_need_stack) {
            lu << inputs[input_id]->get_element_type().c_type_string() << "* volatile * s_input" << input_id << " = (" << inputs[input_id]->get_element_type().c_type_string() << "* volatile *) input" + std::to_string(m_context->inputs.size() - 1) << " + " << m_param_offset[inputs[input_id]->get_name()] << ";\n";
        }
        for (auto output_id: output_need_stack) {
            lu << outputs[output_id]->get_element_type().c_type_string() << "* volatile * s_output" << output_id << " = (" << outputs[output_id]->get_element_type().c_type_string() << "* volatile *) input" + std::to_string(m_context->inputs.size() - 1) << " + " << m_param_offset[outputs[output_id]->get_name()] << ";\n";
        }
    } else {
        lu << "__shared__ int s_label[" << FLAGS_frecursive_max_depth << "];\n";
        for (auto input_id: input_need_stack) {
            lu << "__shared__ " << inputs[input_id]->get_element_type().c_type_string() << "* s_input" << input_id << "[" << FLAGS_frecursive_max_depth << "];\n";
        }
        for (auto output_id: output_need_stack) {
            lu << "__shared__ " << outputs[output_id]->get_element_type().c_type_string() << "* s_output" << output_id << "[" << FLAGS_frecursive_max_depth << "];\n";
        }
        if (FLAGS_fparallel_recursion) {
            lu << "__shared__ int s_local_block_size[" << FLAGS_frecursive_max_depth << "];\n";
            lu << "__shared__ int s_local_block_id[" << FLAGS_frecursive_max_depth << "];\n";
        }
    }
    lu << "int stack_top = 1;\n";
    if (FLAGS_fstack_in_glb)
        lu << "bool is_master_thread = (threadIdx.x == 0 && blockIdx.x == 0);\n";
    else
        lu << "bool is_master_thread = (threadIdx.x == 0);\n";
    lu << "if (is_master_thread)\n";
    lu.block_begin();
    lu << "s_label[stack_top] = 0;\n";
    for (auto input_id: input_need_stack) {
        lu << "s_input" << input_id << "[stack_top] = input" << input_id << ";\n";
    }
    for (auto output_id: output_need_stack) {
        lu << "s_output" << output_id << "[stack_top] = output" << output_id << ";\n";
    }
    if (FLAGS_fparallel_recursion) {
        lu << "s_local_block_size[stack_top] = gridDim.x;\n";
        lu << "s_local_block_id[stack_top] = blockIdx.x;\n";
    }
    lu.block_end();
    if (FLAGS_fstack_in_glb)
        lu << "Barrier();\n";
    else
        lu << "__syncthreads();\n";
    lu << "while (stack_top)";
    lu.block_begin();
    lu << "func_start:;\n";
    lu << "int label = s_label[stack_top];\n";
    for (auto input_id: input_need_stack) {
        lu << "input" << input_id << " = s_input" << input_id << "[stack_top];\n";
    }
    for (auto output_id: output_need_stack) {
        lu << "output" << output_id << " = s_output" << output_id << "[stack_top];\n";
    }
    if (FLAGS_fparallel_recursion) {
        lu << "int local_block_id = s_local_block_id[stack_top];\n";
        lu << "int local_block_size = s_local_block_size[stack_top];\n";
    }
    if (FLAGS_fstack_in_glb)
        lu << "Barrier();\n";
    else
        lu << "__syncthreads();\n";
    lu << "switch (label)";
    lu.block_begin();
    for (int i = 0; i < call_sites.size(); i++) {
        lu << "case " << i+1 << ": goto LABEL" << i+1 << "; break;\n"; 
    }
    lu << "default: break;\n";
    lu.block_end();
    NNFUSION_LOG(INFO) << "call_sites";
    for (int i = 0; i < call_sites.size(); i++) std::cout << call_sites[i] << std::endl;
    // HACK: assume all call sites can run in parallel. check is needed.
    std::cout <<" ------------------ " << std::endl;
    for (int i = 0; i < call_sites.size(); i++) {
        LanguageUnit_p _lu_call(new LanguageUnit(get_function_name() + "_to_stack_" + std::to_string(i)));
        LanguageUnit& lu_call = *_lu_call;
        lu_call.indent = 2;
        lu_call << "if (is_master_thread)";
        lu_call.block_begin();
        if (i == 0 && FLAGS_fparallel_recursion) {
            lu_call << "if (" << FLAGS_fparallel_recursion_min << " * " << call_sites.size() << " <= local_block_size)";
            lu_call.block_begin();
            lu_call << "int block_per_call = local_block_size / " << call_sites.size() << ";\n";
            lu_call << "int remain = local_block_size % " << call_sites.size() << ";\n";
            lu_call << "int start_block_id = 0;\n";
            lu_call << "int end_block_id, new_local_block_size;\n";
            for (int j = 0; j < call_sites.size(); j++) {
                lu_call << "new_local_block_size = " << j << " < remain ? block_per_call + 1 : block_per_call;\n";
                lu_call << "end_block_id = start_block_id + new_local_block_size;\n";
                lu_call << "if (local_block_id >= start_block_id && local_block_id < end_block_id)\n";
                lu_call.block_begin();
                lu_call << "s_label[stack_top] = " << call_sites.size() << ";\n";
                lu_call << "stack_top++;\n";
                for (auto input_id: input_need_stack) {
                    if (input_id != m_context->inputs.size() - 1) {
                        lu_call << "s_input" << input_id << "[stack_top] = " << caller_params[j][input_id] << ";\n";
                    } else {
                        lu_call << "s_input" << input_id << "[stack_top] = " << caller_params[j][input_id] << " + start_block_id / " << FLAGS_fparallel_recursion_min << " * " << FLAGS_frecursive_max_depth << " * " << m_workspace_size << ";\n";
                    }
                }
                for (auto output_id: output_need_stack) {
                    lu_call << "s_output" << output_id << "[stack_top] = " << caller_params[j][output_id + inputs.size()] << ";\n";
                }
                lu_call << "s_label[stack_top] = 0;\n";
                lu_call << "s_local_block_size[stack_top] = new_local_block_size;\n";
                lu_call << "s_local_block_id[stack_top] = local_block_id - start_block_id;\n";
                lu_call.block_end();
                lu_call << "start_block_id = end_block_id;\n";
            }
            lu_call.block_end(); 
            lu_call << "else\n";
            lu_call.block_begin();
        }
        lu_call << "s_label[stack_top] = " << i + 1 << ";\n";
        lu_call << "stack_top++;\n";
        for (auto input_id: input_need_stack) {
            lu_call << "s_input" << input_id << "[stack_top] = " << caller_params[i][input_id] << ";\n";
        }
        for (auto output_id: output_need_stack) {
            lu_call << "s_output" << output_id << "[stack_top] = " << caller_params[i][output_id + inputs.size()] << ";\n";
        }
        lu_call << "s_label[stack_top] = 0;\n";
        lu_call << "s_local_block_size[stack_top] = local_block_size;\n";
        lu_call << "s_local_block_id[stack_top] = local_block_id;\n";
        lu_call.block_end();
        if (i == 0 && FLAGS_fparallel_recursion) lu_call.block_end();
        lu_call << "else stack_top++;\n";
        if (FLAGS_fstack_in_glb)
            lu_call << "Barrier();\n";
        else
            lu_call << "__syncthreads();\n";
        lu_call << "goto func_start;\n";
        lu_call << "LABEL" << i + 1 << ":;\n";
        code = replace_one(code, call_sites[i], lu_call.get_code());
    }
    lu << code;
    lu << "stack_top--;\n";
    lu.block_end();
    return lu.get_code();
}

void cuda::Recursion::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    auto instructions = m_body_instructions;
    for (auto ins : *instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        lu << get_launch_bound(ins);
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
            params.push_back(m_param_map[tensor]);
        for (auto tensor : ins->get_outputs())
            params.push_back(m_param_map[tensor]);
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(m_param_map[tensor]);
        if (FLAGS_frecursive_stack && is_emitting_block_kernel) {
            lu << "// " << kernel->emit_block_kernel_call(params)->get_code();
            std::string inlined = inline_kernel(ins);
            // todo: to_stack should be called per recursive op
            std::string stacked = to_stack(inlined, ins);
            lu << stacked;
        } else {
            lu << kernel->emit_block_kernel_call(params)->get_code();
        }
        if (FLAGS_ffast_barrier) {
            if (dynamic_pointer_cast<ControlFlowEmitter>(kernel) == nullptr && dynamic_pointer_cast<FuncForward>(kernel) == nullptr)
                lu << "Barrier();\n";
        } else {
            if (ins != instructions->back())
                lu << "Barrier();\n";
        }
    }
}

LanguageUnit_p cuda::Recursion::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
    if (FLAGS_frecursive_stack && !is_emitting_block_kernel) {
        std::vector<std::string> params;
        for (size_t i = 0; i < m_context->inputs.size(); i++) {
            stringstream ss;
            ss << "input" << i;
            params.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++) {
            stringstream ss;
            ss << "output" << i;
            params.push_back(ss.str());
        }
        if (FLAGS_ffast_barrier) {
            params.push_back("be_state_buffer");
            params.push_back("state_base");
        }
        params.push_back("shared_buffer");
        lu << m_block_func_name << "_block_kernel(" << join(params, ", ") << ");";
    } else {
        generate_subgraph_code(_lu);
    }
    return _lu;
}

void cuda::Recursion::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_body_instructions);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
    if (FLAGS_fparallel_recursion_grid != -1) m_gridDim.x = FLAGS_fparallel_recursion_grid;
}

static void replace_dep_code(LanguageUnit_p lu, const std::string& src, const std::string& tgt)
{
    auto code = lu->get_code();
    lu->code_symbol_replace(src, tgt);
    for (auto item : lu->local_symbol)
        replace_dep_code(item.second, src, tgt);
}

LanguageUnit_p cuda::Recursion::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    if (FLAGS_fparallel_recursion) _lu->require(declaration::block_barrier);
    if (FLAGS_ffast_barrier) _lu->require(declaration::step_to_device);
    auto saved = m_kernel_name;
    m_kernel_name = m_block_func_name;
    // include the recursion kernel declare at first
    auto kernel_declare = this->emit_device_function_signature();
    (*kernel_declare) << ";\n";
    _lu->require(kernel_declare);
    LanguageUnit_p kernel_code(new LanguageUnit(get_function_name() + "_block_kernel"));
    (*kernel_code) << this->emit_device_function_signature()->get_code();
    kernel_code->block_begin();
    is_emitting_block_kernel = true;
    (*kernel_code) << emit_function_body()->get_code();
    is_emitting_block_kernel = false;
    kernel_code->block_end();
    m_kernel_name = saved;

    for (auto ins : *m_body_instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        // HACK: still need to emit the if kernel when inlined, for emiting the kernels the ``if'' depends on
        _lu->require(block_kernel);
        kernel_code->require(block_kernel);
    }

    _lu->require(kernel_code);
    replace_dep_code(_lu, "$stack_size$", std::to_string(m_workspace_size));
    return _lu;
}

REGISTER_KERNEL_EMITTER("FuncForward",                                             // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::FuncForward)
REGISTER_KERNEL_EMITTER("Recursion",                                               // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Recursion)
