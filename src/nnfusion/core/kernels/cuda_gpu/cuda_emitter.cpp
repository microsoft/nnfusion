// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_int32(fmax_block_dim, 256, "Max blockDim for cuda kernel");
DEFINE_int32(fmax_grid_dim, 256, "Max blockDim for cuda kernel");

LanguageUnit_p cuda::CudaEmitter::emit_function_call()
{
    vector<string> names;
    //set stream during codegen
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());

    return CudaEmitter::emit_function_call(names);
}

LanguageUnit_p cuda::CudaEmitter::emit_function_call(std::vector<std::string> names)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    set_launch_config();

    auto gnode = m_context->gnode;
    string stream_name = "0";
    if (gnode && (*gnode)["Async_info"].is_valid())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        if (async_info.execution_stream != nullptr)
            stream_name = async_info.execution_stream->get_name();
    }
    
    lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
       << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, " << stream_name
       << ">>>(" << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_function_signature()
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

    set_launch_config();
    emit_function_body();
    lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
       << ") __global__ void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_block_kernel()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel"));
    auto& lu = *_lu;

    FunctionUnit_p fu = this->get_or_emit_source();
    lu << fu->comment_unit->get_code();
    lu << this->emit_device_function_signature()->get_code() << "\n";
    lu.block_begin();
    lu << this->emit_device_function_body()->get_code();
    lu.block_end();

    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_block_kernel_call(std::vector<std::string> params)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_call"));
    auto& lu = *_lu;
    params.push_back("shared_buffer");
    lu << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";
    return _lu;
}

LanguageUnit_p cuda::BlockCudaEmitter::emit_block_kernel_call(std::vector<std::string> params)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_call"));
    auto& lu = *_lu;
    params.push_back("threadIdx.x");
    params.push_back("blockIdx.x");
    params.push_back("shared_buffer");
    lu << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";
    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_device_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_body"));
    auto& lu = *_lu;
    FunctionUnit_p fu = this->get_or_emit_source();
    lu << fu->body_unit->get_code() << "\n";
    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_device_function_signature()
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
    params.push_back("char* shared_buffer");
    lu << "__device__ __noinline__ void " << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ")";
    return _lu;
}

shared_ptr<nnfusion::cache::KernelEntry>
    cuda::CudaEmitter::get_kernel_cache_entry(shared_ptr<nnfusion::cache::KernelEntry> kernel_entry)
{
    if (kernel_entry == nullptr)
    {
        kernel_entry = std::make_shared<nnfusion::cache::KernelEntry>();
    }
    FunctionUnit_p func_p = this->get_or_emit_source();
    if (func_p == nullptr)
    {
        NNFUSION_LOG(ERROR) << "Cannot generate kernel_cache_entry due to invalid KernelEmitter: "
                            << m_context->gnode->get_name();
        return nullptr;
    }

    if (kernel_entry->device_type == "")
    {
        kernel_entry->device_type = "CUDA_GPU";
    }

    if (kernel_entry->function.find("grid_dim") == kernel_entry->function.end())
    {
        kernel_entry->function["grid_dim"] = {m_gridDim.x, m_gridDim.y, m_gridDim.z};
    }
    if (kernel_entry->function.find("block_dim") == kernel_entry->function.end())
    {
        kernel_entry->function["block_dim"] = {m_blockDim.x, m_blockDim.y, m_blockDim.z};
    }

    kernel_entry->tags.insert("CudaEmitter");

    return KernelEmitter::get_kernel_cache_entry(kernel_entry);
}

LanguageUnit_p cuda::BlockCudaEmitter::emit_device_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_sig"));
    auto& lu = *_lu;
    LanguageUnit_p _sig(emit_function_signature());
    std::string sig_code = _sig->get_code();
    size_t param_start = sig_code.find("void (") + 6;
    std::string param_str = sig_code.substr(param_start, sig_code.find_last_of(')') - param_start);

    vector<string> params;
    params.push_back("int thread_id");
    params.push_back("int block_id");
    params.push_back("char *shared_buffer");

    lu << "__device__ __noinline__ void " << m_kernel_name << "_block_kernel"
       << "(" << param_str << ", " << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::BlockCudaEmitter::emit_device_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_body"));
    auto& lu = *_lu;

    int block_size = m_blockDim.x * m_blockDim.y * m_blockDim.z;
    int block_num = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    is_emitting_block_kernel = true;
    FunctionUnit_p fu = this->get_or_emit_source();
    is_emitting_block_kernel = false;

    lu << "if (thread_id >= " << block_size << ")";
    lu.block_begin();
    if (num_local_thread_sync > 0)
    {
        lu << "for (int i = 0; i < " << num_local_thread_sync << "; i++) __syncthreads();\n";
    }
    lu << "return;\n";
    lu.block_end();

    lu << "const dim3 blockDim(" << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z
       << ");\n";
    lu << "const dim3 gridDim(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z
       << ");\n";

    if (m_blockDim.y != 1 && m_blockDim.z == 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", thread_id / "
           << m_blockDim.x << ", 0);\n";
    }
    else if (m_blockDim.y == 1 && m_blockDim.z != 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", 0, thread_id / "
           << m_blockDim.x << ");\n";
    }
    else if (m_blockDim.y != 1 && m_blockDim.z != 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", thread_id / "
           << m_blockDim.x << " % " << m_blockDim.y << ", thread_id / "
           << m_blockDim.x * m_blockDim.y << ");\n";
    }

    if (m_gridDim.y == 1 && m_gridDim.z == 1)
    {
        lu << "const dim3 blockIdx(block_id, 0, 0);\n";
    }
    else if (m_gridDim.z == 1)
    {
        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / " << m_gridDim.x
           << ", 0);\n";
    }
    else
    {
        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / " << m_gridDim.x
           << " % " << m_gridDim.y << ", block_id / " << m_gridDim.x * m_gridDim.y << ");\n";
    }

    lu << fu->body_unit->get_code() << "\n";

    return _lu;
}

const std::unordered_map<std::string, size_t> cuda::BlockCudaEmitter::size_of_str_type{
    {"char", sizeof(char)},
    {"float", sizeof(float)},
    {"double", sizeof(double)},
    {"int8_t", sizeof(int8_t)},
    {"int16_t", sizeof(int16_t)},
    {"int32_t", sizeof(int32_t)},
    {"int64_t", sizeof(int64_t)},
    {"uint8_t", sizeof(uint8_t)},
    {"uint16_t", sizeof(uint16_t)},
    {"uint32_t", sizeof(uint32_t)},
    {"uint64_t", sizeof(uint64_t)}};

LanguageUnit_p cuda::AntaresCudaKernelEmitter::emit_function_body()
{
    GENERIC_OP_LOGGING();
    if (antares_code.empty())
    {
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    if (kernel_info.size() > 1)
    {
        std::string to_search = "template_op";
        std::string replace_str = get_function_name();
        int search_pos = antares_code.find(to_search);
        while (search_pos > 0)
        {
            antares_code.replace(search_pos, to_search.size(), replace_str);
            search_pos = antares_code.find(to_search, search_pos + replace_str.size());
        }

        std::vector<int> kernels_pos;
        std::string start = "// LOCAL: ";
        int pos = antares_code.find(start, 0);
        while (pos > 0)
        {
            kernels_pos.push_back(pos);
            pos = antares_code.find(start, pos + start.size());
        }
        NNFUSION_CHECK(kernels_pos.size() == kernel_info.size());
        for (size_t i = 0; i < kernels_pos.size(); i++)
        {
            std::string kernel;
            if (i == kernels_pos.size() - 1)
            {
                kernel = antares_code.substr(kernels_pos[i]);
            }
            else
            {
                kernel = antares_code.substr(kernels_pos[i], kernels_pos[i + 1] - kernels_pos[i]);
            }

            std::string code_start = "extern \"C\" __global__";

            std::string code_end = "\n}\n";
            int p_code_start = kernel.find(code_start);
            int p_code_end = kernel.find(code_end);
            NNFUSION_CHECK(p_code_start >= 0 && p_code_end >= 0 && p_code_end > p_code_start);
            std::string kernel_def = kernel.substr(p_code_start, p_code_end - p_code_start + 3);
            LanguageUnit_p kernel_i = std::make_shared<LanguageUnit>(
                get_function_name() + "_kernel" + to_string(i), kernel_def);
            _lu->require(kernel_i);

            dim3 GridDim, BlockDim;
            find_launch_config(kernel_def, GridDim, BlockDim);
            std::string call;
            auto ki = kernel_info[i];
            NNFUSION_CHECK(ki->kernel_name == "template_op_kernel" + to_string(i));
            call = get_function_name() + "_kernel" + to_string(i) + "<<<dim3(" +
                   to_string(GridDim.x) + ", " + to_string(GridDim.y) + ", " +
                   to_string(GridDim.z) + "), dim3(" + to_string(BlockDim.x) + ", " +
                   to_string(BlockDim.y) + ", " + to_string(BlockDim.z) + "), mem, stream>>>(" +
                   join(ki->input_names, ", ") + ", " + join(ki->output_names, ", ") + ");";
            lu << call << "\n";
        }
    }
    else
    {
        // extract kernel code
        int start = antares_code.find(") {\n"), end = antares_code.find("\n}\n");
        NNFUSION_CHECK(start >= 0 && end >= 0 && end > start);
        std::string str = antares_code.substr(start + 4, end - start - 4);

        find_launch_config(str, m_gridDim, m_blockDim);
        // lu.block_begin();
        lu << str << "\n";
        // lu.block_end();
    }
    return _lu;
}

LanguageUnit_p cuda::AntaresCudaKernelEmitter::emit_function_call()
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

    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());

    if (kernel_info.size() > 1)
    {
        lu << "(0, " << stream_name << ", " << join(names, ", ") << ");\n";
    }
    else
    {
        lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
           << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, "
           << stream_name << ">>>(" << join(names, ", ") << ");\n";
    }
    return _lu;
}

LanguageUnit_p cuda::AntaresCudaKernelEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    std::unordered_set<int> inplace_input, inplace_output;
    if (m_context->annotations)
    {
        for (auto oi_pair : m_context->annotations->get_in_place_oi_pairs())
        {
            inplace_input.insert(oi_pair.input);
            inplace_output.insert(oi_pair.output);
        }
    }
    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        if (inplace_input.find(i) == inplace_input.end())
        {
            ss << "__restrict__ ";
        }
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        if (inplace_output.find(i) == inplace_output.end())
        {
            ss << "__restrict__ ";
        }
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        ss << "__restrict__ ";
        ss << "mediate" << i;
        params.push_back(ss.str());
    }

    set_launch_config();
    emit_function_body();
    if (kernel_info.size() > 1)
    {
        lu << "extern void (unsigned mem, cudaStream_t stream, " << join(params, ", ") << ")";
    }
    else
    {
        lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
           << ") __global__ void "
           << "(" << join(params, ", ") << ")";
    }
    return _lu;
}

void cuda::AntaresCudaKernelEmitter::process_antares_kernel_info()
{
    for (auto ki : kernel_info)
    {
        for (size_t i = 0; i < ki->input_names.size(); i++)
        {
            std::string name = ki->input_names[i];
            if (tensor_name_map.find(name) == tensor_name_map.end())
            {
                if (name.find("input") != std::string::npos)
                {
                    int idx = std::atoi(name.substr(5).data());
                    tensor_name_map[name] = m_context->input_names[idx];
                }
                else if (name.find("mediate") != std::string::npos)
                {
                    std::string dtype_str = ki->input_dtypes[i];
                    element::Type dtype;
                    NNFUSION_CHECK(
                        element::Type::dtype_string_to_nnfusion_element_type(dtype_str, dtype));

                    std::string shape_str = ki->input_shapes[i].substr(1);
                    std::vector<size_t> shape;
                    shape.push_back(std::atoi(shape_str.data()));
                    int pos = shape_str.find(", ");
                    while (pos > 0)
                    {
                        shape_str = shape_str.substr(pos + 2);
                        shape.push_back(std::atoi(shape_str.data()));
                        pos = shape_str.find(", ");
                    }

                    auto tmp_tensor = allocate_tensor(Shape(shape), dtype);
                    tensor_name_map[name] = tmp_tensor->get_name();
                }
            }
        }

        for (size_t i = 0; i < ki->output_names.size(); i++)
        {
            std::string name = ki->output_names[i];
            if (tensor_name_map.find(name) == tensor_name_map.end())
            {
                if (name.find("output") != std::string::npos)
                {
                    int idx = std::atoi(name.substr(6).data());
                    tensor_name_map[name] = m_context->output_names[idx];
                }
                else if (name.find("mediate") != std::string::npos)
                {
                    std::string dtype_str = ki->output_dtypes[i];
                    element::Type dtype;
                    NNFUSION_CHECK(
                        element::Type::dtype_string_to_nnfusion_element_type(dtype_str, dtype));

                    std::string shape_str = ki->output_shapes[i].substr(1);
                    std::vector<size_t> shape;
                    shape.push_back(std::atoi(shape_str.data()));
                    int pos = shape_str.find(", ");
                    while (pos > 0)
                    {
                        shape_str = shape_str.substr(pos + 2);
                        shape.push_back(std::atoi(shape_str.data()));
                        pos = shape_str.find(", ");
                    }

                    auto tmp_tensor = allocate_tensor(Shape(shape), dtype);
                    tensor_name_map[name] = tmp_tensor->get_name();
                }
            }
        }
    }
}

void cuda::AntaresCudaKernelEmitter::find_launch_config(const std::string& str,
                                                        dim3& gridDim,
                                                        dim3& blockDim)
{
    int at_bx = str.find("// [thread_extent] blockIdx.x = "),
        blockX =
            (at_bx >= 0)
                ? std::atoi(str.data() + at_bx + sizeof("// [thread_extent] blockIdx.x = ") - 1)
                : 1;
    int at_by = str.find("// [thread_extent] blockIdx.y = "),
        blockY =
            (at_by >= 0)
                ? std::atoi(str.data() + at_by + sizeof("// [thread_extent] blockIdx.y = ") - 1)
                : 1;
    int at_bz = str.find("// [thread_extent] blockIdx.z = "),
        blockZ =
            (at_bz >= 0)
                ? std::atoi(str.data() + at_bz + sizeof("// [thread_extent] blockIdx.z = ") - 1)
                : 1;
    int at_tx = str.find("// [thread_extent] threadIdx.x = "),
        threadX =
            (at_tx >= 0)
                ? std::atoi(str.data() + at_tx + sizeof("// [thread_extent] threadIdx.x = ") - 1)
                : 1;
    int at_ty = str.find("// [thread_extent] threadIdx.y = "),
        threadY =
            (at_ty >= 0)
                ? std::atoi(str.data() + at_ty + sizeof("// [thread_extent] threadIdx.y = ") - 1)
                : 1;
    int at_tz = str.find("// [thread_extent] threadIdx.z = "),
        threadZ =
            (at_tz >= 0)
                ? std::atoi(str.data() + at_tz + sizeof("// [thread_extent] threadIdx.z = ") - 1)
                : 1;

    gridDim = dim3(blockX, blockY, blockZ);
    blockDim = dim3(threadX, threadY, threadZ);
}

shared_ptr<nnfusion::cache::KernelEntry> cuda::BlockCudaEmitter::get_kernel_cache_entry(
    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry)
{
    if (kernel_entry == nullptr)
    {
        kernel_entry = shared_ptr<nnfusion::cache::KernelEntry>();
    }
    FunctionUnit_p func_p = this->get_or_emit_source();
    if (func_p == nullptr)
    {
        NNFUSION_LOG(ERROR) << "Cannot generate kernel_cache_entry due to invalid KernelEmitter: "
                            << m_context->gnode->get_name();
        return nullptr;
    }

    // only support kernels without shared_memory and local_thread_sync yet
    {
        std::string function_body = func_p->body_unit->get_code();
        if (function_body.find("__shared__") != std::string::npos ||
            function_body.find("__syncthreads") != std::string::npos)
        {
            NNFUSION_LOG(INFO) << "BlockCudaEmitter::get_kernel_cache_entry only supports "
                                  "kernels without shared_memory and "
                                  "local_thread_sync yet, fallback to CudaEmitter";
            return CudaEmitter::get_kernel_cache_entry(kernel_entry);
        }
    }

    if (kernel_entry->function.find("num_syncthreads") == kernel_entry->function.end())
    {
        kernel_entry->function["num_syncthreads"] = num_local_thread_sync;
    }
    if (kernel_entry->function.find("shared_memory") == kernel_entry->function.end())
    {
        kernel_entry->function["shared_memory"]["symbol"] = shared_memory_log.symbol;
        kernel_entry->function["shared_memory"]["dtype"] = shared_memory_log.dtype;
        kernel_entry->function["shared_memory"]["size"] = shared_memory_log.size;
    }
    if (kernel_entry->function.find("block_function_body") == kernel_entry->function.end())
    {
        // TODO(lingm): process function_body with ply to extract shared_memory
        kernel_entry->function["block_function_body"] = func_p->body_unit->get_code();
    }

    kernel_entry->tags.insert("BlockCudaEmitter");

    return CudaEmitter::get_kernel_cache_entry(kernel_entry);
}

LanguageUnit_p cuda::AntaresCudaKernelEmitter::emit_dependency()
{
    GENERIC_OP_LOGGING();

    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

bool cuda::AntaresCudaKernelEmitter::is_eliminative()
{
    return (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]));
}

shared_ptr<nnfusion::cache::KernelEntry> cuda::AntaresCudaKernelEmitter::get_kernel_cache_entry(
    shared_ptr<nnfusion::cache::KernelEntry> kernel_entry)
{
    if (kernel_entry == nullptr)
    {
        kernel_entry = std::make_shared<nnfusion::cache::KernelEntry>();
    }
    FunctionUnit_p func_p = this->get_or_emit_source();
    if (func_p == nullptr)
    {
        NNFUSION_LOG(ERROR) << "Cannot generate kernel_cache_entry due to invalid KernelEmitter: "
                            << m_context->gnode->get_name();
        return nullptr;
    }

    if (kernel_entry->source == "")
    {
        kernel_entry->source = "Antares";
    }

    kernel_entry->miscs["antares"]["time"] =
        nnfusion::kernels::AntaresKEImp::get_perf(this->antares_code) * 1000000; // sec to ms
    kernel_entry->miscs["antares"]["device_name"] =
        nnfusion::kernels::AntaresKEImp::get_device_name(this->antares_code);
    auto tuning_step = nnfusion::kernels::AntaresKEImp::get_tuning_step(this->antares_code);
    kernel_entry->miscs["antares"]["step_produced"] = tuning_step.first;
    kernel_entry->miscs["antares"]["planned_steps"] = tuning_step.second;

    // kernel_entry->miscs["antares"]["antares_response"] = this->antares_code;

    return BlockCudaEmitter::get_kernel_cache_entry(kernel_entry);
}

LanguageUnit_p cuda::CacheCudaEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    std::stringstream ss;
    ss.str(kernel_entry.function["function_signature"]);
    lu << ss.str();

    return _lu;
}

LanguageUnit_p cuda::CacheCudaEmitter::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    std::stringstream ss;
    ss.str(kernel_entry.function["function_body"]);
    lu << ss.str();

    return _lu;
}

LanguageUnit_p cuda::CacheCudaEmitter::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    // Todo: load dependency from kernel cache
    // *_lu << kernel_entry.function["function_dep"];
    _lu->require(header::cuda);
    return _lu;
}

void cuda::CacheCudaEmitter::set_launch_config()
{
    auto func = kernel_entry.function;
    m_gridDim = dim3(func["grid_dim"][0], func["grid_dim"][1], func["grid_dim"][2]);
    m_blockDim = dim3(func["block_dim"][0], func["block_dim"][1], func["block_dim"][2]);
}

LanguageUnit_p cuda::CacheBlockCudaEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    std::stringstream ss;
    ss.str(kernel_entry.function["function_signature"]);
    lu << ss.str();

    return _lu;
}

LanguageUnit_p cuda::CacheBlockCudaEmitter::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto func = kernel_entry.function;

    NNFUSION_CHECK(func.find("shared_memory") != func.end());
    if (func["shared_memory"].size() > 0)
    { // Todo: offload the code conversion effort to users
        for (size_t i = 0; i < func["shared_memory"]["symbol"].size(); i++)
        {
            emit_alloc_shared(lu,
                              func["shared_memory"]["symbol"][i],
                              func["shared_memory"]["dtype"][i],
                              func["shared_memory"]["size"][i]);
        }
    }

    NNFUSION_CHECK(func.find("num_syncthreads") != func.end());
    num_local_thread_sync = func["num_syncthreads"];

    lu.block_begin();
    std::stringstream ss;
    ss.str(func["block_function_body"]);
    lu << ss.str() << "\n";
    lu.block_end();
    return _lu;
}

LanguageUnit_p cuda::CacheBlockCudaEmitter::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    // Todo: load dependency from kernel cache
    // *_lu << kernel_entry.function["function_dep"];
    _lu->require(header::cuda);
    return _lu;
}

void cuda::CacheBlockCudaEmitter::set_launch_config()
{
    auto func = kernel_entry.function;
    m_gridDim = dim3(func["grid_dim"][0], func["grid_dim"][1], func["grid_dim"][2]);
    m_blockDim = dim3(func["block_dim"][0], func["block_dim"][1], func["block_dim"][2]);
}
