// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"
#include <set>
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_int32(fcf_level);
DECLARE_int32(fmax_grid_dim);
DEFINE_int32(floop_copy_blockdim, 256, "");
DECLARE_bool(ffast_barrier);
DECLARE_string(fdefault_device);

cuda::Loop::Loop(shared_ptr<KernelContext> ctx, size_t reserve_memory, int input_output_index_bias)
    : ControlFlowEmitter(ctx), m_input_output_index_bias(input_output_index_bias)
{
    std::stringstream tag;
    tag << "_LoopOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::Loop>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_loop_body_tu = op->get_loop_body_tu();
    size_t workspace_size = 0;
    for (auto& pair : m_loop_body_tu->memory_allocator_factory->get_allocator_list())
    {
        m_pool_offset[pair.second->get_name()] = workspace_size;
        workspace_size += pair.second->max_allocated();
    }
    reserved_memory_start = workspace_size;
    m_workspace = allocate_tensor(Shape{workspace_size + reserve_memory}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    auto output_map = op->get_loop_output_map();
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    for (auto& item : output_map)
        item.second--;
    m_body_instructions = create_param_map(m_loop_body_tu->program, output_map, FLAGS_fcf_level==1);
    if (FLAGS_ffast_barrier) {
        set_launch_config();
        m_sync_tensor = std::make_shared<nnfusion::descriptor::Tensor>(
                    nnfusion::element::i32,
                    nnfusion::PartialShape(
                        {(size_t)m_gridDim.x * (size_t)m_gridDim.y * (size_t)m_gridDim.z}),
                        get_function_name() + "_be_state_buffer",
                        nnfusion::get_device_type(FLAGS_fdefault_device));
        m_sync_tensor->set_memset(true, 0);
        m_sync_tensor->set_persistent(true);
        m_context->tensors.push_back(m_sync_tensor);
        m_context->tensor_names.push_back(m_sync_tensor->get_name());
    }
    m_input_output_index_bias = input_output_index_bias;
}

void fetch_dependent(const std::set<int64_t>& emitted, std::vector<std::shared_ptr<nnfusion::graph::GNode>>& depend, std::shared_ptr<nnfusion::graph::GNode> node) {
    for (auto edge: node->get_in_edges()) {
        auto src = edge->get_src();
        if (emitted.find(src->get_id()) == emitted.end()) {
            fetch_dependent(emitted, depend, src);
        } else {
            depend.push_back(src);
        }
    }
}

bool cuda::Loop::is_host_kernel_launch() {
    return FLAGS_fcf_level == 1;
}

LanguageUnit_p cuda::Loop::emit_function_signature() {
    if (FLAGS_fcf_level == 2) {
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
    return ControlFlowEmitter::emit_function_signature();
}

void cuda::Loop::generate_subgraph_code(LanguageUnit_p _lu, bool in_cuda)
{
    auto& lu = *_lu;
    std::set<int64_t> outputs;
    std::set<int64_t> emitted;
    bool all_kernel_uses_single_block = true;
    for (auto ins : *m_body_instructions)
    {
        auto node = ins->getGNode();
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        // std::cout << node->get_id() << " " << node->get_op_type() << " depends on: ";
        if (in_cuda) {
            // whether to add a barrier
            // bool need_barrier = false;
            // std::vector<std::shared_ptr<nnfusion::graph::GNode>> depend;
            // fetch_dependent(emitted, depend, node);
            // for (auto src: depend) {
            //     if (outputs.find(src->get_id()) != outputs.end()) {
            //         need_barrier = true;
            //         // std::cout << "(depend)";
            //     }
            //     // std::cout << src->get_id() << " " << src->get_op_type() << ", ";
            // }
            // // std::cout << std::endl;
            bool need_barrier = true;
            if (need_barrier && ins != m_body_instructions->front()) {
                if (FLAGS_ffast_barrier && std::dynamic_pointer_cast<ControlFlowEmitter>(kernel) != nullptr) {
                    NNFUSION_CHECK_FAIL() << "TODO: skip barrier for control flow kernels";
                }
                if (all_kernel_uses_single_block && (kernel->get_grid_dim().x == 1 && kernel->get_grid_dim().y == 1 && kernel->get_grid_dim().z == 1)) {
                    lu << "if (blockIdx.x == 0) { __threadfence(); __syncthreads(); }\n";
                } else {
                    lu << "Barrier();\n";
                }
                all_kernel_uses_single_block = true;
                outputs.clear();
            }
            outputs.insert(node->get_id());
            emitted.insert(node->get_id());
        }
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
            params.push_back(m_param_map[tensor]);
        for (auto tensor : ins->get_outputs())
            params.push_back(m_param_map[tensor]);
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(m_param_map[tensor]);
        if (in_cuda) {
            cuda::dim3 grid = kernel->get_grid_dim();
            std::string launch_bound = get_launch_bound(ins);
            std::string kernel_call = kernel->emit_block_kernel_call(params)->get_code();
            if (grid.x > FLAGS_fmax_grid_dim) {
                kernel_call = replace_one(kernel_call, "blockIdx.x", "blockIdx.x + start_block");
                launch_bound = replace_one(launch_bound, "blockIdx.x", "blockIdx.x + start_block");
                lu << "for (int start_block = 0; start_block < " << grid.x << "; start_block += gridDim.x)\n";
                lu.block_begin();
                lu << launch_bound;
                lu.block_begin();
                lu << kernel_call;
                lu.block_end();
                lu.block_end();
            } else {
                lu << launch_bound;
                lu << kernel_call;
            }
        } else {
            if (dynamic_pointer_cast<ControlFlowEmitter>(kernel) != nullptr) {
                auto control_flow_kernel = static_pointer_cast<ControlFlowEmitter>(kernel);
                if (control_flow_kernel->is_host_kernel_launch()) {
                    std::string func_name = control_flow_kernel->get_function_name();
                    std::vector<std::string> param_pointers;
                    dim3 grid_dim = control_flow_kernel->get_grid_dim();
                    dim3 block_dim = control_flow_kernel->get_block_dim();
                    if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::CUDA_GPU) {
                        std::string hashed_func_name = "tmp_" + std::to_string(std::hash<std::string>{}(func_name));
                        int param_id = 0;
                        for (auto param: params) {
                            lu << "void* " << hashed_func_name << "_" << std::to_string(param_id) << " = " << param << ";\n";
                            param_pointers.push_back("&" + hashed_func_name + "_" + std::to_string(param_id));
                            param_id ++;
                        }
                        lu << "void* " << func_name << "_param[] = {" << join(param_pointers, ", ") << "};\n";
                        lu << "cudaLaunchCooperativeKernel((const void*)" << func_name << ", " 
                        << "dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), "
                        << "dim3(" << block_dim.x << "," << block_dim.y << ", " << block_dim.z << "), "
                        << func_name << "_param, (size_t) 0, (cudaStream_t) 0);\n";

                    } else if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::ROCM_GPU) {
                        lu << "void* " << func_name << "_param[] = {" << join(param_pointers, ", ") << "};\n";
                        lu << func_name;
                        lu << "<<<dim3(" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << "), dim3("
                        << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << "), 0, 0>>>(" << join(params, ", ") << ");\n";
                    } else {
                        NNFUSION_CHECK_FAIL() << "Unsupported device: " << FLAGS_fdefault_device;
                    }
                } else {
                    lu << kernel->get_function_name() << kernel->emit_function_call(params)->get_code();
                }
            } else {
                lu << kernel->get_function_name() << kernel->emit_function_call(params)->get_code();
            }
        }
        all_kernel_uses_single_block &= (kernel->get_grid_dim().x == 1 && kernel->get_grid_dim().y == 1 && kernel->get_grid_dim().z == 1);
    }
}

LanguageUnit_p cuda::Loop::emit_function_call()
{
    if (FLAGS_fcf_level == 1) {
        return CudaEmitter::emit_function_call();
    } else {
        return KernelEmitter::emit_function_call();
    }
}

LanguageUnit_p cuda::Loop::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    if (FLAGS_fcf_level == 2) {
        lu << "int64_t iter = 0;\n";
        lu << "CUDA_SAFE_CALL(cudaMemcpy(&iter, input0, sizeof(int64_t), cudaMemcpyDeviceToHost));\n";
        lu << "CUDA_SAFE_CALL(cudaMemcpy(&iter, input0, sizeof(int64_t), cudaMemcpyDeviceToHost));\n";
        lu << "int64_t* i_dev;\n";
        lu << "CUDA_SAFE_CALL(cudaMalloc((void**)&i_dev, sizeof(int64_t)));\n";
        lu << "CUDA_SAFE_CALL(cudaMemset(i_dev, 0, sizeof(int64_t)));\n";
        bool need_copy = false;
        for (auto ins: *m_body_instructions) {
            std::string func_name = ins->getKernel()->get_function_name();
            if (func_name.find("ScatterND") != std::string::npos) {
                need_copy = true;
                break;
            }
        }
        if (!need_copy)
            lu << "if (iter == 0)";
        lu.block_begin();
        int grid_dim = 0;
        for (int i = 0; i < m_context->outputs.size(); i++) {
            size_t tensor_size = shape_size(m_context->outputs[i]->get_shape());
            grid_dim += ceil_div(tensor_size, (size_t) FLAGS_floop_copy_blockdim);
        }
        std::string copy_func_name = get_function_name() + "_copy_kernel";
        vector<string> params_with_type;
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "input" << i + 2;
            params_with_type.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "output" << i;
            params_with_type.push_back(ss.str());
        }
        lu << copy_func_name << "<<<" << "dim3(" << grid_dim << ", 1, 1), dim3(" << FLAGS_floop_copy_blockdim << ", 1, 1)" << ", 0, 0>>>(" << join(params_with_type, ", ") << ");\n";
        lu.block_end();
        if (need_copy) {
            for (int i = 0; i < m_context->outputs.size(); i++)
                lu << "input" << i + 2 << " = output" << i << ";\n";
        }
        lu << "for (int64_t i = 0; i < iter; i++)";
        lu.block_begin();
        generate_subgraph_code(_lu, false);
        lu << "inc_iter<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, 0>>>(i_dev);\n";
        for (int i = 0; i < m_context->outputs.size(); i++)
            lu << "input" << i + 2 << " = output" << i << ";\n";
        lu.block_end();
    } else {
        allocate_shared_memory(_lu);
        lu << "int tid=threadIdx.x + blockIdx.x * blockDim.x;\n";
        for (int i = 0; i < m_context->outputs.size(); i++)
        {
            size_t tensor_size = m_context->outputs[i]->size(false);
            size_t num_threads = m_blockDim.x * m_gridDim.x;
            lu << "for (int64_t i=tid; i<" << tensor_size << "; i+=" << num_threads << ")";
            lu << " output" << i << "[i] = input" << i + 2 << "[i];\n";
        }
        for (int i = 0; i < m_context->outputs.size(); i++)
            lu << "input" << i + 2 << " = output" << i << ";\n";
        lu << "Barrier();\n";
        lu << "for (int64_t i = 0; i < *input0; i++)";
        lu.block_begin();
        lu << "Barrier();\n";
        generate_subgraph_code(_lu, true);
        lu.block_end();
    }
    return _lu;
}

void cuda::Loop::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_body_instructions);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
    m_gridDim.x = min(m_gridDim.x, FLAGS_fmax_grid_dim);
}

LanguageUnit_p cuda::Loop::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::CUDA_GPU)
        _lu->require(declaration::barrier);
    else if (nnfusion::get_device_type(FLAGS_fdefault_device) == nnfusion::NNFusion_DeviceType::ROCM_GPU)
        _lu->require(declaration::manual_barrier);
    else NNFUSION_CHECK_FAIL() << "Unknown device type: " << FLAGS_fdefault_device;
    if (FLAGS_ffast_barrier) _lu->require(declaration::step_to_device);
    for (auto ins : *m_body_instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        if (FLAGS_fcf_level == 2) {
            LanguageUnit_p _k(new LanguageUnit(kernel->get_function_name() + "_loop"));
            auto& k = *_k;
            if (std::dynamic_pointer_cast<cuda::ControlFlowEmitter>(kernel)) {
                vector<string> params;
                vector<string> params_with_type;
                for (size_t i = 0; i < kernel->m_context->inputs.size(); i++)
                {
                    stringstream ss;
                    ss << "input" << i;
                    params.push_back(ss.str());
                }

                for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
                {
                    stringstream ss;
                    ss << "output" << i;
                    params.push_back(ss.str());
                }
                for (size_t i = 0; i < kernel->m_context->inputs.size(); i++)
                {
                    stringstream ss;
                    ss << kernel->m_context->inputs[i]->get_element_type().c_type_string() << "* ";
                    ss << "input" << i;
                    params_with_type.push_back(ss.str());
                }

                for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
                {
                    stringstream ss;
                    ss << kernel->m_context->outputs[i]->get_element_type().c_type_string() << "* ";
                    ss << "output" << i;
                    params_with_type.push_back(ss.str());
                }
                std::string func_name = kernel->get_function_name();
                auto cf_kernel = static_pointer_cast<cuda::ControlFlowEmitter>(kernel);
                if (cf_kernel->is_host_kernel_launch()) {
                    k << "__global__ void " << func_name << "(" << join(params_with_type, ", ") << ")";
                } else {
                    k << "void " << func_name << "(" << join(params_with_type, ", ") << ")";
                }
            } else {
                LanguageUnit_p sig = kernel->emit_function_signature();
                std::string sig_code = sig->get_code();
                size_t param_start = sig_code.find("void (") + 6;
                std::string param_str = sig_code.substr(param_start - 1, sig_code.find_last_of(')') - param_start + 2);
                std::string call_str = sig_code.substr(0, param_start - 1);
                std::string func_name = kernel->get_function_name();
                k << call_str << func_name << param_str << "\n";
            }

            k.block_begin();
            auto func_body = kernel->emit_function_body();
            k << func_body->get_code();
            k.block_end();
            _k->require(kernel->emit_comments());
            _k->copy_require_from(*func_body);
            _k->require(kernel->emit_dependency());
            _lu->require(_k);
            _lu->require(declaration::inc_iter);
        } else {
            auto block_kernel = kernel->emit_block_kernel();
            block_kernel->require(body->dep_unit);
            _lu->require(block_kernel);
        }
    }
    if (FLAGS_fcf_level == 2) {
        std::string copy_func_name = get_function_name() + "_copy_kernel";
        vector<string> params_with_type;
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << m_context->inputs[i + m_input_output_index_bias]->get_element_type().c_type_string() << "* ";
            ss << "input" << i + m_input_output_index_bias;
            params_with_type.push_back(ss.str());
        }

        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
            ss << "output" << i;
            params_with_type.push_back(ss.str());
        }

        LanguageUnit_p _k(new LanguageUnit(get_function_name() + "_loop"));
        auto& k = *_k;
        k << "__global__ void " << copy_func_name << "(" << join(params_with_type, ", ") << ")\n";
        int block_begin = 0;
        k.block_begin();
        for (int i = 0; i < m_context->outputs.size(); i++) {
            size_t tensor_size = shape_size(m_context->outputs[i]->get_shape());
            int block_end = block_begin + ceil_div(tensor_size, (size_t) FLAGS_floop_copy_blockdim);
            if (i > 0) k << "else ";
            k << "if (blockIdx.x >= " << block_begin << " && blockIdx.x < " << block_end << ")\n";
            k.block_begin();
            k << "int tid = (blockIdx.x - " << block_begin << ") * blockDim.x + threadIdx.x;\n";
            k << "if (tid < " << tensor_size << ") output" << i << "[tid] = input" << i + m_input_output_index_bias << "[tid];\n";
            k.block_end();
            block_begin = block_end;
        }
        k.block_end();
        _lu->require(_k);
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER("Loop",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Loop)
