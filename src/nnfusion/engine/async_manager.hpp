// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace async
    {
        class AsyncManager;
        class CUDAAsyncManager;
        class CPUAsyncManager;
        class Stream;
        class Event;
        struct AsyncExecutionInfo;
        class AsyncManagerFactory;
    }
}

class nnfusion::async::Stream
{
public:
    friend class nnfusion::async::AsyncManager;
    bool is_default_stream() const { return m_symbol == "default"; }
    size_t get_stream_id() const { return m_stream_id; }
    DeviceType get_device_type() const { return m_device_type; }
    size_t get_device_id() const { return m_device_id; }
    const std::string& get_device_name() const { return m_device_name; }
    const std::string& get_name() const { return m_name; };
    const std::string& get_symbol() const { return m_symbol; }
private:
    Stream(size_t stream_id, DeviceType device_type, size_t device_id, const string& symbol);
    size_t m_stream_id;
    DeviceType m_device_type;
    size_t m_device_id;
    std::string m_device_name;
    std::string m_name;
    std::string m_symbol;
};

class nnfusion::async::Event
{
public:
    friend class AsyncManager;
    friend class CUDAAsyncManager;
    size_t get_event_id() const { return m_event_id; }
    const shared_ptr<Stream>& get_stream() const { return m_stream; }
    const shared_ptr<nnfusion::op::Op>& get_op() const { return m_op; }
    DeviceType get_device_type() const { return m_stream->get_device_type(); }
    size_t get_device_id() const { return m_stream->get_device_id(); }
    const std::string& get_device_name() const { return m_stream->get_device_name(); }
    const std::string& get_name() const { return m_name; }
    const std::string& get_symbol() const { return m_symbol; }
    bool is_recorded() const { return m_recorded; }
private:
    Event(size_t event_id,
          const shared_ptr<Stream>& stream,
          const shared_ptr<nnfusion::op::Op>& op,
          const string& symbol);
    void set_recorded(bool value = true) { m_recorded = value; }
    size_t m_event_id;
    shared_ptr<Stream> m_stream;
    shared_ptr<nnfusion::op::Op> m_op;
    std::string m_name;
    std::string m_symbol;
    bool m_recorded;
};

class nnfusion::async::AsyncManager
{
public:
    friend class AsyncManagerFactory;
    // create a new stream if not exists, return one if exists.
    shared_ptr<Stream> set_stream(size_t device_id = 0, const string& symbol = "default");
    // create a new event if not exists, return one if exists.
    shared_ptr<Event> set_event(const shared_ptr<Stream>& stream,
                                const shared_ptr<nnfusion::op::Op>& op,
                                const string& symbol = "");
    int num_stream() const { return m_stream_list.size(); }
    int num_event() const { return m_event_list.size(); }
    DeviceType get_device_type() const { return m_device_type; }
    virtual LanguageUnit_p emit_stream_decl();
    virtual LanguageUnit_p emit_event_decl();
    virtual LanguageUnit_p emit_stream_init();
    virtual LanguageUnit_p emit_event_init();
    virtual LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) = 0;
    virtual LanguageUnit_p emit_event_record(shared_ptr<Event> event) = 0;
    virtual LanguageUnit_p emit_event_reset();
    virtual LanguageUnit_p emit_stream_join();
    virtual LanguageUnit_p emit_stream_destroy();
    virtual LanguageUnit_p emit_event_destroy();

protected:
    AsyncManager(DeviceType device_type);
    std::unordered_map<std::string, shared_ptr<Stream>> m_stream_list;
    std::unordered_map<std::string, shared_ptr<Event>> m_event_list;
    DeviceType m_device_type;
};

class nnfusion::async::CUDAAsyncManager : nnfusion::async::AsyncManager
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

private:
    CUDAAsyncManager()
        : AsyncManager(CUDA_GPU)
    {
    }
};

class nnfusion::async::CPUAsyncManager : nnfusion::async::AsyncManager
{
public:
    friend class AsyncManagerFactory;
    LanguageUnit_p emit_event_decl() override;
    LanguageUnit_p emit_event_wait(shared_ptr<Stream> stream, shared_ptr<Event> event) override;
    LanguageUnit_p emit_event_record(shared_ptr<Event> event) override;
    LanguageUnit_p emit_event_reset() override;
    LanguageUnit_p emit_stream_join() override;

private:
    CPUAsyncManager()
        : AsyncManager(GENERIC_CPU)
    {
    }
};

class nnfusion::async::AsyncManagerFactory
{
public:
    AsyncManagerFactory() {}
    static AsyncManager* get_async_manager(DeviceType device_type);
    static const std::unordered_map<std::string, AsyncManager*>& get_async_manager_list()
    {
        return m_async_manager;
    }

private:
    static std::unordered_map<std::string, AsyncManager*> m_async_manager;
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
    shared_ptr<Stream> execution_stream;
    shared_ptr<Event> record_event;
    vector<shared_ptr<Event>> wait_events;
};