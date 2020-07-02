#include "nnfusion/engine/async_manager.hpp"

DECLARE_string(fdefault_device);

using namespace nnfusion::async;

Stream::Stream(size_t stream_id, DeviceType device_type, size_t device_id, const string& symbol)
    : m_stream_id(stream_id)
    , m_device_type(device_type)
    , m_device_id(device_id)
    , m_symbol(symbol)
    , m_name(symbol + "_stream_" + std::to_string(stream_id))
{
    std::string dt_name = (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[device_type];
    m_device_name = dt_name + "_" + std::to_string(device_id);
}

Event::Event(size_t event_id,
             const shared_ptr<Stream>& stream,
             const shared_ptr<nnfusion::op::Op>& op,
             const string& symbol)
    : m_event_id(event_id)
    , m_stream(stream)
    , m_op(op)
    , m_symbol(symbol)
    , m_name(symbol + "_" + op->get_unique_name() + "_event_" + std::to_string(event_id))

{
}

AsyncManager::AsyncManager(DeviceType device_type)
    : m_device_type(device_type)
{
}

// create a new stream if not exists, return one if exists.
shared_ptr<Stream> AsyncManager::set_stream(size_t device_id, const string& symbol)
{
    std::string search_name = std::to_string(device_id) + "_" + symbol + "_stream";
    if (m_stream_list.find(search_name) != m_stream_list.end())
    {
        auto stream = m_stream_list[search_name];
        return m_stream_list[search_name];
    }
    else
    {
        size_t stream_id = m_stream_list.size();
        shared_ptr<Stream> stream(new Stream(stream_id, m_device_type, device_id, symbol));
        m_stream_list[search_name] = stream;
        return stream;
    }
}
// create a new event if not exists, return one if exists.
shared_ptr<Event> AsyncManager::set_event(const shared_ptr<Stream>& stream,
                                          const shared_ptr<nnfusion::op::Op>& op,
                                          const string& symbol)
{
    std::string search_name = stream->get_name() + symbol + "_" + op->get_unique_name() + "_event";
    if (m_event_list.find(search_name) != m_event_list.end())
    {
        return m_event_list[search_name];
    }
    else
    {
        size_t event_id = m_event_list.size();
        shared_ptr<Event> event(new Event(event_id, stream, op, symbol));
        m_event_list[search_name] = event;
        return event;
    }
}

LanguageUnit_p AsyncManager::emit_stream_decl()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_decl"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(new LanguageUnit("event_decl"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_stream_init()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_init"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_event_init()
{
    LanguageUnit_p _lu(new LanguageUnit("event_init"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_event_reset()
{
    LanguageUnit_p _lu(new LanguageUnit("event_reset"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_stream_join()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_join"));
    return _lu;
}

LanguageUnit_p AsyncManager::emit_stream_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_destroy"));
    return _lu;
}
LanguageUnit_p AsyncManager::emit_event_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit("event_destroy"));
    return _lu;
}

// emit code for declaring all cuda streams.
LanguageUnit_p CUDAAsyncManager::emit_stream_decl()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_decl"));
    auto& lu = *_lu;
    for (auto stream_pair : m_stream_list)
    {
        if (stream_pair.second->get_symbol() != "default")
        {
            lu << "cudaStream_t " << stream_pair.second->get_name() << ";\n";
        }
    }
    return _lu;
}

// emit code for declaring all cuda events.
LanguageUnit_p CUDAAsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(new LanguageUnit("event_decl"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << "cudaEvent_t " << event_pair.second->get_name() << ";\n";
    }
    return _lu;
}

// emit code for initializing all cuda streams.
LanguageUnit_p CUDAAsyncManager::emit_stream_init()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_init"));
    auto& lu = *_lu;
    for (auto stream_pair : m_stream_list)
    {
        // Cuda default stream(0) need not to be created.
        if (stream_pair.second->get_symbol() != "default")
        {
            lu << "cudaStreamCreate(&" << stream_pair.second->get_name() << ");\n";
        }
    }
    return _lu;
}

// emit code for initializing all cuda events.
LanguageUnit_p CUDAAsyncManager::emit_event_init()
{
    LanguageUnit_p _lu(new LanguageUnit("event_init"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << "cudaEventCreateWithFlags(&" << event_pair.second->get_name()
           << ", cudaEventDisableTiming);\n";
    }
    return _lu;
}

// emit code for waiting cuda event. The stream is blocked until the event complete.
LanguageUnit_p CUDAAsyncManager::emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event)
{
    // \todo: we only support stream synchronization on the same device.
    CHECK(stream->get_device_name() == event->get_device_name())
        << "Unsupported event wait operation: synchronize streams on two different devices";
    LanguageUnit_p _lu(new LanguageUnit("event_wait"));
    auto& lu = *_lu;
    if (!event->is_recorded())
        throw nnfusion::errors::RuntimeError("CUDA event error.");
    //LOG(WARNING) << "CUDA event error.";
    if (stream->is_default_stream())
        lu << "cudaStreamWaitEvent(0, " << event->get_name() << ", 0 );\n";
    else
        lu << "cudaStreamWaitEvent(" << stream->get_name() << ", " << event->get_name()
           << ", 0 );\n";
    return _lu;
}

// emit code for declaring all cpu events/notifications.
LanguageUnit_p CPUAsyncManager::emit_event_decl()
{
    LanguageUnit_p _lu(new LanguageUnit("event_decl"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << "nnfusion::cpu::Notification " << event_pair.second->get_name() << ";\n";
    }
    return _lu;
}

// emit code for waiting cpu event/notifications. The stream/thread is blocked until the event complete.
LanguageUnit_p CPUAsyncManager::emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event)
{
    // \todo: we only support stream synchronization on the same device.
    CHECK(stream->get_device_name() == event->get_device_name())
        << "Unsupported event wait operation: synchronize streams on two different devices";
    LanguageUnit_p _lu(new LanguageUnit("event_wait"));
    auto& lu = *_lu;

    lu << event->get_name() << ".Wait();\n";
    return _lu;
}
// emit code for recording cuda event.
LanguageUnit_p CUDAAsyncManager::emit_event_record(shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit("event_record"));
    auto& lu = *_lu;

    if (event->get_stream()->is_default_stream())
        lu << "cudaEventRecord(" << event->get_name() << ", 0);\n";
    else
        lu << "cudaEventRecord(" << event->get_name() << ", " << event->get_stream()->get_name()
           << ");\n";
    event->set_recorded();

    return _lu;
}

// emit code for destroying cuda stream.
LanguageUnit_p CUDAAsyncManager::emit_stream_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_del"));
    auto& lu = *_lu;
    for (auto stream_pair : m_stream_list)
    {
        // Cuda default stream(0) need not to be destroyed.
        if (stream_pair.second->get_symbol() != "default")
        {
            lu << "cudaStreamDestroy(" << stream_pair.second->get_name() << ");\n";
        }
    }
    return _lu;
}
// emit code for destroying cuda event.
LanguageUnit_p CUDAAsyncManager::emit_event_destroy()
{
    LanguageUnit_p _lu(new LanguageUnit("event_del"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        {
            lu << "cudaEventDestroy(" << event_pair.second->get_name() << ");\n";
        }
    }
    return _lu;
}

// emit code for recording cpu event/notification.
LanguageUnit_p CPUAsyncManager::emit_event_record(shared_ptr<Event> event)
{
    LanguageUnit_p _lu(new LanguageUnit("event_record"));
    auto& lu = *_lu;

    lu << event->get_name() << ".Notify();\n";

    return _lu;
}
// emit code for reseting all CPU events/notifications.
LanguageUnit_p CPUAsyncManager::emit_event_reset()
{
    LanguageUnit_p _lu(new LanguageUnit("event_reset"));
    auto& lu = *_lu;
    for (auto event_pair : m_event_list)
    {
        lu << event_pair.second->get_name() << ".Reset();\n";
    }
    return _lu;
}

// emit code for synchronizing all cpu streams/threads.
// It blocks the main/default thread until other threads execution has completed.
LanguageUnit_p CPUAsyncManager::emit_stream_join()
{
    LanguageUnit_p _lu(new LanguageUnit("stream_join"));
    auto& lu = *_lu;
    for (auto stream_pair : m_stream_list)
    {
        if (!stream_pair.second->is_default_stream())
        {
            lu << stream_pair.second->get_name() << ".join();\n";
        }
    }
    return _lu;
}

std::unordered_map<std::string, AsyncManager*> AsyncManagerFactory::m_async_manager;

AsyncManager* AsyncManagerFactory::get_async_manager(DeviceType device_type)
{
    std::string dt_name = (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[device_type];
    if (m_async_manager.find(dt_name) != m_async_manager.end())
    {
        return m_async_manager[dt_name];
    }
    else
    {
        AsyncManager* async_manager = nullptr;
        switch (device_type)
        {
        case CUDA_GPU:
        {
            async_manager = new CUDAAsyncManager();
            break;
        }
        //\ todo: temporirly rocm use cuda's async manager.
        case ROCM_GPU:
        {
            async_manager = new CUDAAsyncManager();
            break;
        }
        case GENERIC_CPU:
        {
            async_manager = new CPUAsyncManager();
            break;
        }
        default: nnfusion::errors::NotSupported("The device is not supported.");
        }
        if (async_manager != nullptr)
            m_async_manager[dt_name] = async_manager;
        return async_manager;
    }
}