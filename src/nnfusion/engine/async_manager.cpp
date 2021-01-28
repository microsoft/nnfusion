// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/engine/async_manager.hpp"

DECLARE_string(fdefault_device);
DECLARE_int32(fnum_non_cpu);
DECLARE_string(fhlsl_codegen_type);

using namespace nnfusion::async;

Stream::Stream(size_t stream_id,
               NNFusion_DeviceType device_type,
               size_t device_id,
               const string& symbol)
    : m_stream_id(stream_id)
    , m_device_type(device_type)
    , m_device_id(device_id)
    , m_symbol(symbol)
{
    if (device_type == GENERIC_CPU)
    {
        m_name = symbol + "_thread";
    }
    else
    {
        if (is_default_stream() && (device_type == CUDA_GPU || device_type == ROCM_GPU ||
                                    FLAGS_fhlsl_codegen_type != "csharp"))
        {
            m_name = "0";
        }
        else if (is_default_stream() && device_type == HLSL)
        {
            m_name = "IntPtr.Zero";
        }
        else
        {
            std::string dev_name = "";
            if (FLAGS_fnum_non_cpu > 1)
                dev_name = "_dev" + to_string(device_id);
            m_name = symbol + dev_name + "_stream";
        }
    }

    std::string dt_name = get_device_str(device_type);
    m_device_name = dt_name + "_" + std::to_string(device_id);
}

void Stream::add_binding_symbol(const std::string& binding_symbol)
{
    NNFUSION_CHECK(binding_symbol != "");
    if (m_binding_symbol.find(binding_symbol) == m_binding_symbol.end())
    {
        std::string content = binding_symbol + "_" + this->get_name();
        m_binding_symbol[binding_symbol] = content;
    }
}

Event::Event(size_t event_id, const shared_ptr<Stream>& stream, const string& symbol)
    : m_event_id(event_id)
    , m_stream(stream)
    , m_symbol(symbol)

{
    if (get_device_type() == GENERIC_CPU)
    {
        m_name = symbol + "_barrier";
    }
    else
    {
        m_name = symbol + "_event";
    }
}

// create a new thread/stream if not exists, return one if exists.
shared_ptr<Stream> AsyncManager::set_stream(size_t device_id, const string& symbol)
{
    std::string search_name = std::to_string(device_id) + "_" + symbol;
    if (m_stream_list.find(search_name) != m_stream_list.end())
    {
        auto stream = m_stream_list[search_name];
        return m_stream_list[search_name];
    }
    else
    {
        size_t stream_id = m_stream_list.size();
        shared_ptr<Stream> stream(new Stream(stream_id, get_device_type(), device_id, symbol));
        m_stream_list[search_name] = stream;
        m_dev_stream[device_id].push_back(stream);

        if (symbol != "default")
            m_num_non_default_stream += 1;

        return stream;
    }
}

// create a new barrier/event if not exists, return one if exists.
shared_ptr<Event> AsyncManager::set_event(const shared_ptr<Stream>& stream, const string& symbol)
{
    std::string search_name = stream->get_name() + symbol + "_event";
    if (m_event_list.find(search_name) != m_event_list.end())
    {
        return m_event_list[search_name];
    }
    else
    {
        size_t event_id = m_event_list.size();
        shared_ptr<Event> event(new Event(event_id, stream, symbol));
        m_event_list[search_name] = event;
        m_dev_event[event->get_device_id()].push_back(event);
        return event;
    }
}

// emit code for declaring all device streams.
LanguageUnit_p DeviceStreamAsyncManager::emit_stream_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_stream_decl"));
    auto& lu = *_lu;
    for (auto stream_pair : m_stream_list)
    {
        if (stream_pair.second->get_symbol() != "default")
        {
            lu << "cudaStream_t " << stream_pair.second->get_name() << ";\n";
        }
        // binding info
        for (auto binding_symbol_pair : stream_pair.second->get_binding_symbol())
        {
            if (binding_symbol_pair.first == "cudnn_handle")
                lu << "cudnnHandle_t " << binding_symbol_pair.second << ";\n";
            else if (binding_symbol_pair.first == "cublas_handle")
                lu << "cublasHandle_t " << binding_symbol_pair.second << ";\n";
            else
                nnfusion::errors::RuntimeError("Unknown stream binding info.");
        }
    }
    return _lu;
}

// emit code for declaring all device events.
LanguageUnit_p DeviceStreamAsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_event_decl"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << "cudaEvent_t " << event_pair.second->get_name() << ";\n";
    }
    return _lu;
}

// emit code for initializing all device streams.
LanguageUnit_p DeviceStreamAsyncManager::emit_stream_init()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_stream_init"));
    auto& lu = *_lu;
    lu << "// create streams/handles\n";
    for (auto& info : m_dev_stream)
    {
        if (info.second.size() > 1 || !info.second[0]->is_default_stream())
            lu << "CUDA_SAFE_CALL(cudaSetDevice(" << info.first << "));\n";
        for (auto stream : info.second)
        {
            // Cuda default stream(0) need not to be created.
            if (!stream->is_default_stream())
            {
                lu << "CUDA_SAFE_CALL(cudaStreamCreate(&" << stream->get_name() << "));\n";
            }
            // binding info
            for (auto binding_symbol_pair : stream->get_binding_symbol())
            {
                if (binding_symbol_pair.first == "cudnn_handle")
                {
                    lu << "CUDNN_SAFE_CALL(cudnnCreate(&" << binding_symbol_pair.second << "));\n";
                    if (!stream->is_default_stream())
                        lu << "CUDNN_SAFE_CALL(cudnnSetStream(" << binding_symbol_pair.second
                           << ", " << stream->get_name() << "));\n";
                }
                else if (binding_symbol_pair.first == "cublas_handle")
                {
                    lu << "CUBLAS_SAFE_CALL(cublasCreate(&" << binding_symbol_pair.second
                       << "));\n";
                    if (!stream->is_default_stream())
                        lu << "CUBLAS_SAFE_CALL(cublasSetStream(" << binding_symbol_pair.second
                           << ", " << stream->get_name() << "));\n";
                }
                else
                    nnfusion::errors::RuntimeError("Unknown stream binding info.");
            }
        }
    }
    return _lu;
}

// emit code for initializing all device events.
LanguageUnit_p DeviceStreamAsyncManager::emit_event_init()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_init"));
    auto& lu = *_lu;
    lu << " // create events\n";
    for (auto info : m_dev_event)
    {
        lu << "CUDA_SAFE_CALL(cudaSetDevice(" << info.first << "));\n";
        for (auto event : info.second)
        {
            lu << "CUDA_SAFE_CALL(cudaEventCreateWithFlags(&" << event->get_name()
               << ", cudaEventDisableTiming));\n";
        }
    }
    return _lu;
}

// emit code for waiting device event. The stream is blocked until the event complete.
LanguageUnit_p DeviceStreamAsyncManager::emit_event_wait(shared_ptr<Stream> stream,
                                                         shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_wait"));
    auto& lu = *_lu;
    if (stream->is_default_stream())
        lu << "CUDA_SAFE_CALL(cudaStreamWaitEvent(0, " << event->get_name() << ", 0 ));\n";
    else
        lu << "CUDA_SAFE_CALL(cudaStreamWaitEvent(" << stream->get_name() << ", "
           << event->get_name() << ", 0));\n";
    return _lu;
}

// emit code for recording device event.
LanguageUnit_p DeviceStreamAsyncManager::emit_event_record(shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_record"));
    auto& lu = *_lu;

    if (event->get_stream()->is_default_stream())
        lu << "CUDA_SAFE_CALL(cudaEventRecord(" << event->get_name() << ", 0));\n";
    else
        lu << "CUDA_SAFE_CALL(cudaEventRecord(" << event->get_name() << ", "
           << event->get_stream()->get_name() << "));\n";
    // event->set_recorded();

    return _lu;
}

// emit code for destroying device stream.
LanguageUnit_p DeviceStreamAsyncManager::emit_stream_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_stream_del"));
    auto& lu = *_lu;
    for (auto& info : m_dev_stream)
    {
        if (info.second.size() > 1 || !info.second[0]->is_default_stream())
            lu << "CUDA_SAFE_CALL(cudaSetDevice(" << info.first << "));\n";
        for (auto stream : info.second)
        {
            // Cuda default stream(0) need not to be destroyed.
            if (stream->get_symbol() != "default")
            {
                lu << "CUDA_SAFE_CALL(cudaStreamDestroy(" << stream->get_name() << "));\n";
            }
            // binding info
            for (auto binding_symbol_pair : stream->get_binding_symbol())
            {
                if (binding_symbol_pair.first == "cudnn_handle")
                {
                    lu << "CUDNN_SAFE_CALL(cudnnDestroy(" << binding_symbol_pair.second << "));\n";
                }
                else if (binding_symbol_pair.first == "cublas_handle")
                {
                    lu << "CUBLAS_SAFE_CALL(cublasDestroy(" << binding_symbol_pair.second
                       << "));\n";
                }
                else
                    nnfusion::errors::RuntimeError("Unknown stream binding info.");
            }
        }
    }
    return _lu;
}
// emit code for destroying device event.
LanguageUnit_p DeviceStreamAsyncManager::emit_event_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_del"));
    auto& lu = *_lu;
    for (auto& info : m_dev_event)
    {
        lu << "CUDA_SAFE_CALL(cudaSetDevice(" << info.first << "));\n";
        for (auto event : info.second)
        {
            lu << "CUDA_SAFE_CALL(cudaEventDestroy(" << event->get_name() << "));\n";
        }
    }
    return _lu;
}

// emit code for declaring all host streams.
LanguageUnit_p HostAsyncManager::emit_stream_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_stream_decl"));
    auto& lu = *_lu;
    return _lu;
}

// emit code for declaring all cpu events/notifications.
LanguageUnit_p HostAsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_event_decl"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << "nnfusion::cpu::Notification " << event_pair.second->get_name() << ";\n";
    }
    return _lu;
}

// emit code for waiting cpu event/notifications. The stream/thread is blocked until the event complete.
LanguageUnit_p HostAsyncManager::emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_wait"));
    auto& lu = *_lu;

    lu << event->get_name() << ".Wait();\n";
    return _lu;
}

// emit code for recording cpu event/notification.
LanguageUnit_p HostAsyncManager::emit_event_record(shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_record"));
    auto& lu = *_lu;

    lu << event->get_name() << ".Notify();\n";

    return _lu;
}
// emit code for reseting all CPU notifications.
LanguageUnit_p HostAsyncManager::emit_event_reset()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_reset"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << event_pair.second->get_name() << ".Reset();\n";
    }
    return _lu;
}

LanguageUnit_p HLSLAsyncManager::emit_stream_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_stream_decl"));
    auto& lu = *_lu;
    return _lu;
}
LanguageUnit_p HLSLAsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(
        new LanguageUnit("declaration::" + get_device_str(m_device_type) + "_event_decl"));
    auto& lu = *_lu;
    return _lu;
}
LanguageUnit_p HLSLAsyncManager::emit_stream_init()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_stream_init"));
    auto& lu = *_lu;
    lu << "// create streams\n";
    for (auto& info : m_dev_stream)
    {
        for (auto stream : info.second)
        {
            // HLSL default stream(0) need not to be created.
            if (!stream->is_default_stream())
            {
                if (FLAGS_fhlsl_codegen_type == "cpp")
                {
                    lu << "auto " << stream->get_name() << " = dxStreamCreate();\n";
                }
                else if (FLAGS_fhlsl_codegen_type == "csharp")
                {
                    lu << "var " << stream->get_name() << " = dxStreamCreate();\n";
                }
            }
        }
    }
    return _lu;
}
LanguageUnit_p HLSLAsyncManager::emit_event_init()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_init"));
    auto& lu = *_lu;
    lu << " // create events\n";
    for (auto info : m_dev_event)
    {
        for (auto event : info.second)
        {
            if (FLAGS_fhlsl_codegen_type == "cpp")
            {
                lu << "auto " << event->get_name() << "= dxEventCreate();\n";
            }
            else if (FLAGS_fhlsl_codegen_type == "csharp")
            {
                lu << "var " << event->get_name() << "= dxEventCreate();\n";
            }
        }
    }
    return _lu;
}

LanguageUnit_p HLSLAsyncManager::emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event)
{
    nnfusion::errors::NotSupported("HLSL async manager does not support event wait api.");
}

LanguageUnit_p HLSLAsyncManager::emit_event_record(shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_record"));
    auto& lu = *_lu;

    if (event->get_stream()->is_default_stream())
    {
        if (FLAGS_fhlsl_codegen_type == "cpp")
        {
            lu << "dxEventRecord(" << event->get_name() << ", 0);\n";
        }
        else if (FLAGS_fhlsl_codegen_type == "csharp")
        {
            lu << "dxEventRecord(" << event->get_name() << ", IntPtr.Zero);\n";
        }
    }
    else
    {
        lu << "dxEventRecord(" << event->get_name() << ", " << event->get_stream()->get_name()
           << ");\n";
    }
    // event->set_recorded();

    return _lu;
}

LanguageUnit_p HLSLAsyncManager::emit_stream_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_stream_del"));
    auto& lu = *_lu;
    for (auto& info : m_dev_stream)
    {
        for (auto stream : info.second)
        {
            // HLSL default stream(0) need not to be destroyed.
            if (stream->get_symbol() != "default")
            {
                lu << "dxStreamDestroy(" << stream->get_name() << ");\n";
            }
        }
    }
    return _lu;
}

LanguageUnit_p HLSLAsyncManager::emit_event_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit(get_device_str(m_device_type) + "_event_del"));
    auto& lu = *_lu;
    for (auto& info : m_dev_event)
    {
        for (auto event : info.second)
        {
            lu << "dxEventDestroy(" << event->get_name() << ");\n";
        }
    }
    return _lu;
}

std::unordered_map<std::string, HostAsyncManager*> AsyncManagerFactory::m_host_async_manager;
std::unordered_map<std::string, DeviceStreamAsyncManager*>
    AsyncManagerFactory::m_device_stream_async_manager;
HostAsyncManager*
    AsyncManagerFactory::get_host_async_manager(std::shared_ptr<nnfusion::graph::Graph> graph,
                                                NNFusion_DeviceType device_type)
{
    NNFUSION_CHECK(device_type == GENERIC_CPU) << "Unsupported device type";
    std::string search_name = get_device_str(device_type);
    if (graph)
        search_name += graph->get_name();
    if (m_host_async_manager.find(search_name) != m_host_async_manager.end())
    {
        return m_host_async_manager[search_name];
    }
    else
    {
        HostAsyncManager* host_async_manager = nullptr;
        host_async_manager = new HostAsyncManager(graph);
        if (host_async_manager != nullptr)
            m_host_async_manager[search_name] = host_async_manager;
        return host_async_manager;
    }
}

DeviceStreamAsyncManager* AsyncManagerFactory::get_device_stream_async_manager(
    std::shared_ptr<nnfusion::graph::Graph> graph, NNFusion_DeviceType device_type)
{
    std::string search_name = get_device_str(device_type);
    if (graph)
        search_name += graph->get_name();
    if (m_device_stream_async_manager.find(search_name) != m_device_stream_async_manager.end())
    {
        return m_device_stream_async_manager[search_name];
    }
    else
    {
        DeviceStreamAsyncManager* device_stream_async_manager = nullptr;
        switch (device_type)
        {
        case CUDA_GPU:
        {
            device_stream_async_manager = new CUDAAsyncManager(graph);
            break;
        }
        //\ todo: temporirly rocm use cuda's async manager.
        case ROCM_GPU:
        {
            device_stream_async_manager = new CUDAAsyncManager(graph);
            break;
        }
        case HLSL:
        {
            device_stream_async_manager = new HLSLAsyncManager(graph);
            break;
        }
        default: nnfusion::errors::NotSupported("Unsupported device stream async manager.");
        }
        if (device_stream_async_manager != nullptr)
            m_device_stream_async_manager[search_name] = device_stream_async_manager;
        return device_stream_async_manager;
    }
}