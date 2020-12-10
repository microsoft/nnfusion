// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "./cache_emitter.hpp"

using namespace nnfusion::kernels::cuda;

LanguageUnit_p CacheBlockCudaKernel::emit_function_signature()
{
    auto func = m_function;
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));

    std::stringstream ss;
    ss.str(func["function_signature"]);
    *_lu << ss.str();

    return _lu;
}

LanguageUnit_p CacheBlockCudaKernel::emit_function_body()
{
    auto& ctx = m_context;
    auto func = m_function;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

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

LanguageUnit_p CacheBlockCudaKernel::emit_dependency()
{
    auto func = m_function;
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    // Todo: load dependency from kernel cache
    // *_lu << func["function_dep"];
    _lu->require(header::cuda);
    return _lu;
}

void CacheBlockCudaKernel::set_launch_config()
{
    auto func = m_function;
    m_gridDim = dim3(func["grid_dim"][0], func["grid_dim"][1], func["grid_dim"][2]);
    m_blockDim = dim3(func["block_dim"][0], func["block_dim"][1], func["block_dim"][2]);
}