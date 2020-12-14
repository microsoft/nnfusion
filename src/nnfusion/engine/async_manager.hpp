// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace async
    {
        class AsyncManager;
        class DeviceStreamAsyncManager;
        class HostAsyncManager;
        class CUDAAsyncManager;
        class HLSLAsyncManager;
        class Stream;
        class Event;
        struct AsyncExecutionInfo;
        class AsyncManagerFactory;
    }
}

// Stream of nnfusion can represent the stream of gpu and the thread of cpu.
// NNfuison stream managed by CPUAsyncManager is thread while by CUDAAsyncManager is stream.
class nnfusion::async::Stream
{
public:
    friend class nnfusion::async::AsyncManager;
    bool is_default_stream() const { return m_symbol == "default"; }
    size_t get_stream_id() const { return m_stream_id; }
    NNFusion_DeviceType get_device_type() const { return m_device_type; }
    size_t get_device_id() const { return m_device_id; }
    const std::string& get_device_name() const { return m_device_name; }
    const std::string& get_name() const { return m_name; };
    const std::string& get_symbol() const { return m_symbol; }
    const std::unordered_map<std::string, std::string>& get_binding_symbol() const
    {
        return m_binding_symbol;
    }
    void add_binding_symbol(const std::string& binding_symbol);

private:
    Stream(size_t stream_id,
           NNFusion_DeviceType device_type,
           size_t device_id,
           const string& symbol);
    size_t m_stream_id;
    NNFusion_DeviceType m_device_type;
    size_t m_device_id;
    std::string m_device_name;
    std::string m_name;
    std::string m_symbol;
    std::unordered_map<std::string, std::string> m_binding_symbol;
};

// Event of nnfusion can represent the event of gpu and the barrier of cpu.
// NNfuison event managed by CPUAsyncManager is barrier while by CUDAAsyncManager is event.
class nnfusion::async::Event
{
public:
    friend class AsyncManager;
    friend class CUDAAsyncManager;
    friend class HLSLAsyncManager;
    size_t get_event_id() const { return m_event_id; }
    const shared_ptr<Stream>& get_stream() const { return m_stream; }
    const shared_ptr<nnfusion::op::Op>& get_op() const { return m_op; }
    NNFusion_DeviceType get_device_type() const { return m_stream->get_device_type(); }
    size_t get_device_id() const { return m_stream->get_device_id(); }
    const std::string& get_device_name() const { return m_stream->get_device_name(); }
    const std::string& get_name() const { return m_name; }
    const std::string& get_symbol() const { return m_symbol; }
private:
    Event(size_t event_id,
          const shared_ptr<Stream>& stream,
          const shared_ptr<nnfusion::op::Op>& op,
          const string& symbol);
    Event(size_t event_id, const shared_ptr<Stream>& stream, const string& symbol);
    size_t m_event_id;
    shared_ptr<Stream> m_stream;
    shared_ptr<nnfusion::op::Op> m_op;
    std::string m_name;
    std::string m_symbol;
};

class nnfusion::async::AsyncManager
{
public:
    friend class AsyncManagerFactory;
    shared_ptr<Stream> set_stream(size_t device_id = 0, const string& symbol = "default");
    shared_ptr<Event> set_event(const shared_ptr<Stream>& stream, const string& symbol = "");
    int num_stream() const { return m_stream_list.size(); }
    int num_non_default_stream() const { return m_num_non_default_stream; }
    int num_event() const { return m_event_list.size(); }
    NNFusion_DeviceType get_device_type() const { return m_device_type; }
    virtual LanguageUnit_p emit_stream_decl() = 0;
    virtual LanguageUnit_p emit_event_decl() = 0;
    virtual LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) = 0;
    virtual LanguageUnit_p emit_event_record(shared_ptr<Event> event) = 0;

protected:
    AsyncManager(std::shared_ptr<nnfusion::graph::Graph> graph)
        : m_num_non_default_stream{0}
    {
    }
    std::unordered_map<std::string, shared_ptr<Stream>> m_stream_list;
    std::unordered_map<int, std::vector<shared_ptr<Stream>>> m_dev_stream;
    std::unordered_map<std::string, shared_ptr<Event>> m_event_list;
    std::unordered_map<int, std::vector<shared_ptr<Event>>> m_dev_event;
    NNFusion_DeviceType m_device_type;
    int m_num_non_default_stream;
};

class nnfusion::async::DeviceStreamAsyncManager : public nnfusion::async::AsyncManager
{
public:
    friend class AsyncManagerFactory;
    LanguageUnit_p emit_stream_decl() override;
    LanguageUnit_p emit_event_decl() override;
    virtual LanguageUnit_p emit_stream_init();
    virtual LanguageUnit_p emit_event_init();
    LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) override;
    LanguageUnit_p emit_event_record(shared_ptr<Event> event) override;
    virtual LanguageUnit_p emit_stream_destroy();
    virtual LanguageUnit_p emit_event_destroy();

protected:
    DeviceStreamAsyncManager(std::shared_ptr<nnfusion::graph::Graph> graph)
        : AsyncManager(graph)
    {
        m_device_type = CUDA_GPU;
    }
};

class nnfusion::async::HostAsyncManager : public nnfusion::async::AsyncManager
{
public:
    friend class AsyncManagerFactory;
    LanguageUnit_p emit_stream_decl() override;
    LanguageUnit_p emit_event_decl() override;
    LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) override;
    LanguageUnit_p emit_event_record(shared_ptr<Event> event) override;
    virtual LanguageUnit_p emit_event_reset();

protected:
    HostAsyncManager(std::shared_ptr<nnfusion::graph::Graph> graph)
        : AsyncManager(graph)
    {
        m_device_type = GENERIC_CPU;
    }
};

class nnfusion::async::CUDAAsyncManager : public nnfusion::async::DeviceStreamAsyncManager
{
public:
    friend class AsyncManagerFactory;

protected:
    CUDAAsyncManager(std::shared_ptr<nnfusion::graph::Graph> graph)
        : DeviceStreamAsyncManager(graph)
    {
        m_device_type = CUDA_GPU;
    }
};

class nnfusion::async::HLSLAsyncManager : public nnfusion::async::DeviceStreamAsyncManager
{
public:
    friend class AsyncManagerFactory;
    LanguageUnit_p emit_stream_decl() override;
    LanguageUnit_p emit_event_decl() override;
    LanguageUnit_p emit_stream_init() override;
    LanguageUnit_p emit_event_init() override;
    LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) override;
    LanguageUnit_p emit_event_record(shared_ptr<Event> event) override;
    LanguageUnit_p emit_stream_destroy() override;
    LanguageUnit_p emit_event_destroy() override;

protected:
    HLSLAsyncManager(std::shared_ptr<nnfusion::graph::Graph> graph)
        : DeviceStreamAsyncManager(graph)
    {
        m_device_type = HLSL;
    }
};

class nnfusion::async::AsyncManagerFactory
{
public:
    AsyncManagerFactory() {}
    static HostAsyncManager* get_host_async_manager(std::shared_ptr<nnfusion::graph::Graph> graph,
                                                    NNFusion_DeviceType device_type = GENERIC_CPU);
    static DeviceStreamAsyncManager*
        get_device_stream_async_manager(std::shared_ptr<nnfusion::graph::Graph> graph,
                                        NNFusion_DeviceType device_type = CUDA_GPU);
    static const std::unordered_map<std::string, HostAsyncManager*>& get_host_async_manager_list()
    {
        return m_host_async_manager;
    }
    static const std::unordered_map<std::string, DeviceStreamAsyncManager*>&
        get_device_stream_async_manager_list()
    {
        return m_device_stream_async_manager;
    }

private:
    static std::unordered_map<std::string, HostAsyncManager*> m_host_async_manager;
    static std::unordered_map<std::string, DeviceStreamAsyncManager*> m_device_stream_async_manager;
};

struct nnfusion::async::AsyncExecutionInfo
{
    /*
        Execution Stream Info
         stream default: ------- kernel#0 ----------------
                                   |
                                   |--> trigger: event#0
                                                 ^ 
                                   wait: event#0 | 
                                                 |
         stream      #1: -------------------- kernel#1---
         Its the stream who wait for the event.

        Design purpose aync allreduce:
         stream: default | copy:d2h | allreduce | copy:h2d
                   |          |          |           |
                  op          |          |           |
                   |->grad    |          |           |
                   |   |--->memcpy(*)    |           |
                   |          |----->SuperScaler     |
                   |          |          |-(option)->memcpy(*)-->grad
                   |          |          |-(option)-------------> |
                  ...         x          x                        |
                   |                                              |
           event:iteration end                                    |
                   |                                              |
               apply_grad<-------(grad ready event)---------------
                   |
               apply_grad_other_0 (Apply grad for other op)
                   |
               apply_grad_other_1
                   |
                   x <----next interation


        (*) means this operation works stimulously with default stream.
        */

    //CPUAsyncInfo
    shared_ptr<Stream> execution_thread;
    shared_ptr<Event> notify_barrier;
    vector<shared_ptr<Event>> wait_barriers;

    //GPUAsyncInfo
    shared_ptr<Stream> execution_stream;
    shared_ptr<Event> record_event;
    vector<shared_ptr<Event>> wait_events;

    bool sync_stream = false; // for cross-device sync
};