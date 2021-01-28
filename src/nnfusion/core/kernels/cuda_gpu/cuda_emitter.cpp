// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

LanguageUnit_p cuda::CudaEmitter::emit_function_call()
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
    params.push_back("int thread_id");
    params.push_back("int block_id");
    params.push_back("char *shared_buffer");

    lu << "__device__ __noinline__ void " << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ")";
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

    // extract kernel code
    int start = antares_code.find(") {\n"), end = antares_code.find("\n}\n");
    NNFUSION_CHECK(start >= 0 && end >= 0 && end > start);
    std::string str = antares_code.substr(start + 4, end - start - 4);

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

    m_gridDim = dim3(blockX, blockY, blockZ);
    m_blockDim = dim3(threadX, threadY, threadZ);

    lu.block_begin();
    lu << str << "\n";
    lu.block_end();
    return _lu;
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
