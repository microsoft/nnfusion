// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Basic Datastructure used in profiling
 * \author wenxh
 */

#include "profiling_runtime.hpp"

using namespace nnfusion::profiler;

bool IProfilingRuntime::execute(const ProfilingContext::Pointer& ke)
{
    auto kctx = ke->kernel->m_context;
    size_t buffer_size = 0;
    for (auto t : kctx->inputs)
        buffer_size += t->size();
    for (auto t : kctx->outputs)
        buffer_size += t->size();

    char* buffer = new char[buffer_size];
    void** inputs = new void*[kctx->inputs.size()];
    void** outputs = new void*[kctx->inputs.size()];

    // Assign all the tensor in the buffer space.
    size_t offset = 0;
    size_t index = 0;
    for (auto t : kctx->inputs)
    {
        inputs[index++] = buffer + offset;
        offset += t->size();
    }
    index = 0;
    for (auto t : kctx->outputs)
    {
        outputs[index++] = buffer + offset;
        offset += t->size();
    }

    bool ret = execute(ke, inputs, outputs);

    delete inputs;
    delete outputs;
    delete buffer;

    return ret;
}

double IProfilingRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

double IProfilingRuntime::execute(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (ke->using_cache)
        //\todo(wenxh): what should be index for the profiling result.
        return nnfusion::profiler::ProfilingCache::profile_timing_result(
            ke, [&]() { return invoke(ke, input, output); }, this->get_name());
    else
        return invoke(ke, input, output);
}