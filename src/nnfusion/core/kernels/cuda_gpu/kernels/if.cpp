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
DEFINE_bool(fif_launch_then_else_naive, false, "launch kernels in both then branch and else branch without fusion");
DEFINE_bool(fif_launch_d2h, false, "launch kernels in both then branch and else branch");
DEFINE_int32(ffused_max_grid, 512, "max griddim in fused kernels");
DECLARE_bool(ffast_barrier);
DECLARE_string(fdefault_device);

cuda::If::If(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    NNFUSION_CHECK((!FLAGS_fif_launch_d2h) || !(FLAGS_fif_launch_then_else));
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
    m_then_branch_instructions = create_param_map(m_then_branch_tu->program, op->get_output_map(), !(FLAGS_fif_launch_d2h || FLAGS_fif_launch_then_else));
    m_pool_offset = else_branch_pool_offset;
    m_else_branch_instructions = create_param_map(m_else_branch_tu->program, op->get_output_map(), !(FLAGS_fif_launch_d2h || FLAGS_fif_launch_then_else));
    if (FLAGS_ffast_barrier) {
        set_launch_config();
        m_sync_tensor = std::make_shared<nnfusion::descriptor::Tensor>(
                    nnfusion::element::i32,
                    nnfusion::PartialShape(
                        {(size_t)m_gridDim.x * (size_t)m_gridDim.y * (size_t)m_gridDim.z}),
                        get_function_name() + "_be_state_buffer",
                    nnfusion::NNFusion_DeviceType::CUDA_GPU);
        m_sync_tensor->set_memset(true, 0);
        m_sync_tensor->set_persistent(true);
        m_context->tensors.push_back(m_sync_tensor);
        m_context->tensor_names.push_back(m_sync_tensor->get_name());
    }
    // Hardcoded fusion rule: fuse all kernels with shm_size=0
    std::vector<std::vector<int>> then_kernel_groups;
    std::vector<std::vector<int>> else_kernel_groups;
    then_kernel_groups.clear();
    for (int i = 0; i < m_then_branch_instructions->size(); i++) {
        size_t shm_size = get_kernel_shared_memory((*m_then_branch_instructions).at(i)->getKernel());
        if (shm_size > 0) {
            if (then_kernel_groups.size() > 0 && then_kernel_groups[then_kernel_groups.size() - 1].size() == 0) {
                then_kernel_groups[then_kernel_groups.size() - 1].push_back(i);
            } else {
                then_kernel_groups.push_back(std::vector<int>({i}));
            }
            then_kernel_groups.push_back(std::vector<int>());
        } else {
            if (then_kernel_groups.size() == 0) {
                then_kernel_groups.push_back(std::vector<int>());
            }
            then_kernel_groups[then_kernel_groups.size() - 1].push_back(i);
        }
    }
    if (then_kernel_groups[then_kernel_groups.size() - 1].size() == 0)
        then_kernel_groups.pop_back();
    NNFUSION_LOG(INFO) << "then groups from size " << m_then_branch_instructions->size();
    for (auto group: then_kernel_groups) {
        for (auto x: group) printf("%d ", x);
        printf("\n");
    }
    else_kernel_groups.clear();
    for (int i = 0; i < m_else_branch_instructions->size(); i++) {
        size_t shm_size = get_kernel_shared_memory((*m_else_branch_instructions).at(i)->getKernel());
        if (shm_size > 0) {
            if (else_kernel_groups[else_kernel_groups.size() - 1].size() == 0) {
                else_kernel_groups[else_kernel_groups.size() - 1].push_back(i);
            } else {
                else_kernel_groups.push_back(std::vector<int>({i}));
            }
            else_kernel_groups.push_back(std::vector<int>());
        } else {
            if (i == 0) {
                else_kernel_groups.push_back(std::vector<int>());
            }
            else_kernel_groups[else_kernel_groups.size() - 1].push_back(i);
        }
    }
    if (else_kernel_groups[else_kernel_groups.size() - 1].size() == 0)
        else_kernel_groups.pop_back();
    NNFUSION_LOG(INFO) << "else groups";
    for (auto group: else_kernel_groups) {
        for (auto x: group) printf("%d ", x);
        printf("\n");
    }
    m_kernel_groups.clear();
    if (FLAGS_fif_launch_d2h || (FLAGS_fif_launch_then_else && FLAGS_fif_launch_then_else_naive)) {
        for (auto group: then_kernel_groups) {
            m_kernel_groups.push_back(make_pair(group, std::vector<int>()));
        }
        for (auto group: else_kernel_groups) {
            m_kernel_groups.push_back(make_pair(std::vector<int>(), group));
        }
    } else if (FLAGS_fif_launch_then_else) {
        int then_group_p = 0, else_group_p = 0;
        while (then_group_p < then_kernel_groups.size() || else_group_p < else_kernel_groups.size()) {
            if (then_group_p >= then_kernel_groups.size()) {
                m_kernel_groups.push_back(make_pair(std::vector<int>(), else_kernel_groups[else_group_p]));
                else_group_p ++;
                continue;
            }
            if (else_group_p >= else_kernel_groups.size()) {
                m_kernel_groups.push_back(make_pair(then_kernel_groups[then_group_p], std::vector<int>()));
                then_group_p ++;
                continue;
            }
            if (is_dense_op_group(m_then_branch_instructions, then_kernel_groups[then_group_p])) {
                m_kernel_groups.push_back(make_pair(then_kernel_groups[then_group_p], std::vector<int>()));
                then_group_p ++;
                continue;
            }
            if (is_dense_op_group(m_else_branch_instructions, else_kernel_groups[else_group_p])) {
                m_kernel_groups.push_back(make_pair(std::vector<int>(), else_kernel_groups[else_group_p]));
                else_group_p ++;
                continue;
            }
            if (then_kernel_groups[then_group_p].size() == 1 && else_kernel_groups[else_group_p].size() == 1) {
                m_kernel_groups.push_back(make_pair(then_kernel_groups[then_group_p], else_kernel_groups[else_group_p]));
                then_group_p ++; else_group_p ++;
                continue;
            }
            if (then_kernel_groups[then_group_p].size() > 1 && else_kernel_groups[else_group_p].size() > 1) {
                m_kernel_groups.push_back(make_pair(then_kernel_groups[then_group_p], else_kernel_groups[else_group_p]));
                then_group_p ++; else_group_p ++;
                continue;
            }
            if (then_group_p <= else_group_p) {
                m_kernel_groups.push_back(make_pair(then_kernel_groups[then_group_p], std::vector<int>()));
                then_group_p ++;
            } else {
                m_kernel_groups.push_back(make_pair(std::vector<int>(), else_kernel_groups[else_group_p]));
                else_group_p ++;
            }
        }
    }
    NNFUSION_LOG(INFO) << "fused groups";
    for (auto group: m_kernel_groups) {
        printf("(then) ");
        for (auto x: group.first) printf("%d ", x);
        printf("(else) ");
        for (auto x: group.second) printf("%d ", x);
        printf("\n");
    }
    if (FLAGS_fif_launch_then_else) {
        m_outer_control_then = "if (*input0)";
        m_outer_control_else = "if (!(*input0))";
    } else {
        m_outer_control_then = "";
        m_outer_control_else = "";
    }
}

bool cuda::If::is_dense_op_group(ir::BasicBlock::Pointer instructions, std::vector<int> inst_id) {
    // TODO: use get_inst_max_shared_memory
    size_t mx = 0;
    for (int i: inst_id)
        mx = max(mx, get_kernel_shared_memory(instructions->at(i)->getKernel()));
    return mx;
}

bool cuda::If::is_host_kernel_launch() {
    return !FLAGS_fif_launch_d2h && !FLAGS_fif_launch_then_else;
}

void cuda::If::generate_branch_fused_kernel(LanguageUnit_p _lu, ir::BasicBlock::Pointer instructions, int max_grid_dim, int start_id, int end_id)
{
    auto& lu = *_lu;
    if (end_id == -1) end_id = instructions->size();
    int last_griddim = -1;
    bool last_need_wait = false;
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
        if (FLAGS_ffast_barrier && last_need_wait) {
            lu << emit_block_executor_instruction_wait_for(last_griddim, min(griddim, max_grid_dim))->get_code();
        }
        last_griddim = min(griddim, max_grid_dim);
        if (griddim > max_grid_dim) {
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
        if (FLAGS_ffast_barrier) {
            if (!(i == end_id - 1 && !is_emitting_block_kernel)) {
                if (std::dynamic_pointer_cast<ControlFlowEmitter>(kernel) == nullptr) {
                    lu << emit_block_executor_instruction_step_to(last_griddim)->get_code();
                    last_need_wait = true;
                } else {
                    last_need_wait = false;
                }
                //     lu << "Barrier();\n";
            }
        } else {
            if (i != end_id - 1)
                lu << "Barrier();\n";
        }
    }
    if (FLAGS_ffast_barrier && is_emitting_block_kernel) {
        lu << emit_block_executor_instruction_wait_for(last_griddim)->get_code();
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

std::string cuda::If::get_wrapper_func_name(int then_start_id, int then_end_id, int else_start_id, int else_end_id) {
    std::string func_name = get_function_name() + "_branch_wrapper_";
    if (then_start_id < then_end_id) {
        func_name = func_name + "then_" + std::to_string(then_start_id) + "_to_" + std::to_string(then_end_id - 1);
        if (else_start_id != else_end_id) func_name += "_";
    }
    if (else_start_id < else_end_id) {
        func_name = func_name + "else_" + std::to_string(else_start_id) + "_to_" + std::to_string(else_end_id - 1);
    }
    return func_name;
}

int cuda::If::get_max_grid_dim(int then_start_id, int then_end_id, int else_start_id, int else_end_id) {
    bool has_barrier = (then_end_id - then_start_id > 1) || (else_end_id - else_start_id > 1);
    return has_barrier ? FLAGS_ffused_max_grid : INT32_MAX;
}

LanguageUnit_p cuda::If::generate_branch_fused_function(int then_start_id, int then_end_id, int else_start_id, int else_end_id) {
    std::string func_name = get_wrapper_func_name(then_start_id, then_end_id, else_start_id, else_end_id);
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
    bool has_barrier = (then_end_id - then_start_id > 1) || (else_end_id - else_start_id > 1);
    int max_grid = get_max_grid_dim(then_start_id, then_end_id, else_start_id, else_end_id);
    emit_function_body();
    lu << "__global__ void " << func_name << "(" << join(params, ", ") << ")";
    lu.block_begin();
    size_t shm_size = max(
        get_inst_max_shared_memory(m_then_branch_instructions, then_start_id, then_end_id),
        get_inst_max_shared_memory(m_else_branch_instructions, else_start_id, else_end_id)
    );
    allocate_shared_memory(_lu, shm_size);
    if (then_start_id < then_end_id) {
        lu << m_outer_control_then;
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_then_branch_instructions, max_grid, then_start_id, then_end_id);
        lu.block_end();
    }
    if (else_start_id < else_end_id) {
        lu << m_outer_control_else;
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_else_branch_instructions, max_grid, else_start_id, else_end_id);
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
    << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), 0, 0>>>(" << join(params, ", ") << ");\n";
}

void cuda::If::emit_branch_wrapper(int then_start_id, int then_end_id, int else_start_id, int else_end_id, LanguageUnit &lu, bool emit_all_args) {
    std::string func_name = get_wrapper_func_name(then_start_id, then_end_id, else_start_id, else_end_id);
    // auto instructions = else_branch ? m_else_branch_instructions : m_then_branch_instructions;
    int grid_x = 1, block_x = 1;
    for (int i = then_start_id; i < then_end_id; i++) {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(m_then_branch_instructions->at(i)->getKernel());
        grid_x = max(grid_x, kernel->get_grid_dim().x);
        block_x = max(block_x, kernel->get_block_dim().x);
    }
    for (int i = else_start_id; i < else_end_id; i++) {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(m_else_branch_instructions->at(i)->getKernel());
        grid_x = max(grid_x, kernel->get_grid_dim().x);
        block_x = max(block_x, kernel->get_block_dim().x);
    }
    grid_x = min(grid_x, get_max_grid_dim(then_start_id, then_end_id, else_start_id, else_end_id));
    cuda::dim3 grid_dim = dim3(grid_x, 1, 1);
    cuda::dim3 block_dim = dim3(block_x, 1, 1);
    if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::CUDA_GPU) {
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
        if (then_start_id + 1 >= then_end_id && else_start_id + 1 >= else_end_id) {
            lu << "CUDA_SAFE_CALL(cudaLaunchKernel(" ;
        } else {
            lu << "CUDA_SAFE_CALL(cudaLaunchCooperativeKernel(";
        }
        lu << "(const void*) " << func_name << ", "
        << "dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), "
        << "dim3(" << block_dim.x << "," << block_dim.y << ", " << block_dim.z << "), "
        << "args, (size_t) 0, (cudaStream_t)0));\n";
    } else if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::ROCM_GPU) {
        std::vector<string> params;
        for (size_t i = 0; i < m_context->inputs.size(); i++)
        {
            stringstream ss;
            ss << "input" << i;
            params.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "output" << i;
            params.push_back(ss.str());
        }
        lu << func_name << "<<<dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), dim3("
        << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), 0, 0>>>(" << join(params, ", ") << ");\n";
    }
}

LanguageUnit_p cuda::If::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    if (FLAGS_fif_launch_then_else || FLAGS_fif_launch_d2h) {
        auto& lu = *_lu;
        if (FLAGS_fif_launch_d2h) {
            bool cond_on_cpu = false;
            auto node = m_context->gnode;
            if (node->hasAttribute("cpu_tensor")) {
                auto cpu_tensors = node->Get<std::vector<int>>("cpu_tensor");
                if (std::find(cpu_tensors.begin(), cpu_tensors.end(), 0) != cpu_tensors.end()) {
                    cond_on_cpu = true;
                }
            }
            if (cond_on_cpu) {
                lu << "char cond = *input0;\n";
            } else {
                lu << "char cond = 0;\n";
                lu << "CUDA_SAFE_CALL(cudaMemcpy(&cond, input0, sizeof(char), cudaMemcpyDeviceToHost));\n";
            }
            lu << "if (cond)";
            lu.block_begin();
            bool emit_all_args = true;
            for (auto group: m_kernel_groups) {
                if (group.first.size() > 0) {
                    if (group.first.size() == 1) {
                        auto ins = m_then_branch_instructions->at(group.first[0]);
                        emit_kernel_wrapper(ins, lu);
                    } else {
                        emit_branch_wrapper(group.first[0], group.first[group.first.size() - 1] + 1, 0, 0, lu, emit_all_args);
                        emit_all_args = false;
                    }
                }
            }
            lu.block_end();
            lu << "else\n";
            lu.block_begin();
            emit_all_args = true;
            for (auto group: m_kernel_groups) {
                if (group.second.size() > 0) {
                    if (group.second.size() == 1) {
                        auto ins = m_else_branch_instructions->at(group.second[0]);
                        emit_kernel_wrapper(ins, lu);
                    } else {
                        emit_branch_wrapper(0, 0, group.second[0], group.second[group.second.size() - 1] + 1, lu, emit_all_args);
                        emit_all_args = false;
                    }
                }
            }
            lu.block_end();
        } else {
            bool emit_all_args = true;
            for (auto group: m_kernel_groups) {
                if (group.first.size() == 1 && group.second.size() == 0) {
                    auto ins = m_then_branch_instructions->at(group.first[0]);
                    emit_kernel_wrapper(ins, lu);
                } else if (group.first.size() == 0 && group.second.size() == 1) {
                    auto ins = m_else_branch_instructions->at(group.second[0]);
                    emit_kernel_wrapper(ins, lu);
                } else {
                    emit_branch_wrapper(
                        group.first.size() == 0 ? 0 : group.first[0],
                        group.first.size() == 0 ? 0 : group.first[group.first.size() - 1] + 1,
                        group.second.size() == 0 ? 0 : group.second[0],
                        group.second.size() == 0 ? 0 : group.second[group.second.size() - 1] + 1,
                        lu, emit_all_args);
                    emit_all_args = false;
                }
            }
        }
    } else {
        auto& lu = *_lu;
        allocate_shared_memory(_lu);
        lu << "if (*input0) ";
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_then_branch_instructions, FLAGS_ffused_max_grid);
        lu.block_end();
        lu << "else ";
        lu.block_begin();
        generate_branch_fused_kernel(_lu, m_else_branch_instructions, FLAGS_ffused_max_grid);
        lu.block_end();
    }
    return _lu;
}

LanguageUnit_p cuda::If::emit_function_call(std::vector<std::string> names) {
    if (!FLAGS_fif_launch_then_else && !FLAGS_fif_launch_d2h) {
        // NNFUSION_CHECK_FAIL() << "should not reach here";
        // branch in cuda
        return ControlFlowEmitter::emit_function_call(names);
    } else {
        LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
        auto& lu = *_lu;
        auto node = m_context->gnode;
        if (node->hasAttribute("cpu_tensor")) {
            auto cpu_tensors = node->Get<std::vector<int>>("cpu_tensor");
            for (auto i: cpu_tensors) names[i] = names[i] + "_cpu";
        }
        lu << "(";
        lu << join(names, ", ") << ");\n";
        return _lu;
    }
}

LanguageUnit_p cuda::If::emit_function_call()
{
    vector<string> names;
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    return emit_function_call(names);
}

void cuda::If::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_then_branch_instructions);
    auto cfg1 = get_subgraph_launch_config(m_else_branch_instructions);
    m_blockDim = maxdim3(cfg0.first, cfg1.first);
    m_gridDim = maxdim3(cfg0.second, cfg1.second);
    NNFUSION_CHECK(m_gridDim.y == 1);
    NNFUSION_CHECK(m_gridDim.z == 1);
    m_gridDim.x = min(m_gridDim.x, FLAGS_ffused_max_grid);
}

LanguageUnit_p cuda::If::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::CUDA_GPU)
        _lu->require(declaration::barrier);
    else if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::ROCM_GPU)
        _lu->require(declaration::manual_barrier);
    else NNFUSION_CHECK_FAIL() << "Unknown device type: " << FLAGS_fdefault_device;
    if (FLAGS_ffast_barrier) _lu->require(declaration::step_to_device);
    auto op = static_pointer_cast<op::If>(m_context->gnode->get_op_ptr());
    m_then_branch_instructions = create_param_map(m_then_branch_tu->program, op->get_output_map(), !(FLAGS_fif_launch_d2h || FLAGS_fif_launch_then_else));
    m_else_branch_instructions = create_param_map(m_else_branch_tu->program, op->get_output_map(), !(FLAGS_fif_launch_d2h || FLAGS_fif_launch_then_else));

    if (FLAGS_fif_launch_then_else || FLAGS_fif_launch_d2h) {
        // std::string outer_control = FLAGS_fif_launch_then_else ? "if (*input0)" : "";
        for (auto& group: m_kernel_groups) {
            if ((group.first.size() > 0 && group.second.size() > 0) || group.first.size() > 1 || group.second.size() > 1) {
                auto group_kernel = generate_branch_fused_function(
                    group.first.size() == 0 ? 0 : group.first[0],
                    group.first.size() == 0 ? 0 : group.first[group.first.size() - 1] + 1,
                    group.second.size() == 0 ? 0 : group.second[0],
                    group.second.size() == 0 ? 0 : group.second[group.second.size() - 1] + 1
                );
                for (auto inst_id: group.first) {
                    auto ins = m_then_branch_instructions->at(inst_id);
                    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                    auto body = kernel->get_or_emit_source();
                    auto block_kernel = kernel->emit_block_kernel();
                    block_kernel->require(body->dep_unit);
                    group_kernel->require(block_kernel);
                }
                for (auto inst_id: group.second) {
                    auto ins = m_else_branch_instructions->at(inst_id);
                    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                    auto body = kernel->get_or_emit_source();
                    auto block_kernel = kernel->emit_block_kernel();
                    block_kernel->require(body->dep_unit);
                    group_kernel->require(block_kernel);
                }
                _lu->require(group_kernel);
            } // TODO: impl the below two cases in generate_branch_fused_function
            else if (group.first.size() == 1) {
                auto ins = m_then_branch_instructions->at(group.first[0]);
                auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                auto body = kernel->get_or_emit_source();
                auto block_kernel = kernel->emit_block_kernel();
                block_kernel->require(body->dep_unit);
                _lu->require(block_kernel);
                auto ins_kernel = generate_branch_seperate_function(m_outer_control_then, m_context->inputs[0], ins);
                ins_kernel->require(block_kernel);
                _lu->require(ins_kernel);
            } else if (group.second.size() == 1) {
                 auto ins = m_else_branch_instructions->at(group.second[0]);
                auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
                auto body = kernel->get_or_emit_source();
                auto block_kernel = kernel->emit_block_kernel();
                block_kernel->require(body->dep_unit);
                _lu->require(block_kernel);
                auto ins_kernel = generate_branch_seperate_function(m_outer_control_else, m_context->inputs[0], ins);
                ins_kernel->require(block_kernel);
                _lu->require(ins_kernel);        
            } else {
                NNFUSION_CHECK_FAIL() << "unreachable";
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
    if (!FLAGS_fif_launch_then_else && !FLAGS_fif_launch_d2h) return ControlFlowEmitter::emit_function_signature();
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

    set_launch_config();
    emit_function_body();
    lu << "void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}


REGISTER_KERNEL_EMITTER("If",                                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::If)
