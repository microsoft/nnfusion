// Microsoft (c) 2019, NNFusion Team
#include "assign_async_info_pass.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace nnfusion::async;

DECLARE_bool(fadd_allreduce);
DECLARE_string(fdefault_device);
DECLARE_bool(frt_const_folding);
DEFINE_bool(fmulti_stream, false, "Enable multi-stream.");

AssignAsyncInfoPass::AssignAsyncInfoPass()
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    m_device = nnfusion::get_device_type(dev_name);
}

bool AssignAsyncInfoPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto async_manager = AsyncManagerFactory::get_async_manager(m_device);
    assign_stream_info(async_manager, graph);
    assign_event_info(async_manager, graph);
    LOG(INFO) << "run async-------------------------------";
    return true;
}

void AssignAsyncInfoPass::assign_stream_info(AsyncManager* async_manager, shared_ptr<Graph>& graph)
{
    // Four streams in use:
    // default; d2h; h2d; allreduce;
    // allreduce waits for default stream reduce_start;

    bool allreduce_enable = FLAGS_fadd_allreduce;
    bool enable_rt_const_folding = FLAGS_frt_const_folding;
    bool enable_multi_stream = FLAGS_fmulti_stream;
    //const
    set<string> constant_vals;
    for (auto gnode : graph->get_nodes())
    {
        // Detect those operators
        // 1). ApplyGradient
        //     |
        //      ----> (memcpy-DtoH)(AG)(memcpy-HtoD)
        //     memcpy will assign to D2H or H2D stream.
        //     AG will assign to default stream, and wait *ALL* gradient
        //     ready signal. (Or iteration end signal?)
        // 2). AllReduce
        //     |
        //      ----> (memcpy-HtoD)(AR)(memcpy-DtoH)
        //     memcpy will assign to H2D or H2D stream.
        //     AR will assign to its own stream.
        //     It will wait for Host Data ready signal.
        //     It will trigger Reduced signal.
        // For other memcpy operators, no stream assigned.
        (*gnode)["Async_info"] = AsyncExecutionInfo();
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        if (!enable_multi_stream)
        {
            async_info.execution_stream = async_manager->set_stream(0, "default");
        }
        // \todo: stream-assign logic will be revised later.
        else
        {
            // all constant ops use default stream
            if (gnode->get_op_type() == "Constant")
            {
                async_info.execution_stream = async_manager->set_stream(0, "default");
                if (enable_rt_const_folding)
                {
                    for (size_t i = 0; i < gnode->get_output_size(); ++i)
                    {
                        auto tensor = gnode->get_output_tensor_ptr(i);
                        constant_vals.insert(tensor->get_name());
                    }
                }
            }
            // set applygradient stream
            else if (allreduce_enable && gnode->get_op_type() == "AllReduce")
            {
                async_info.execution_stream = async_manager->set_stream(0, "allreduce");
            }
            else
            {
                async_info.execution_stream = async_manager->set_stream(0, "base");
            }

            // If enable runtime constant folding, for cuda codegen, ops whose inputs are all constants are taken as constant ops.
            // And will be called in init() instead of kernel_entry(). So these ops use default stream as well.
            auto dt = async_manager->get_device_type();
            if (enable_rt_const_folding && (dt == CUDA_GPU || dt == ROCM_GPU))
            {
                bool const_inputs = true;
                if (!gnode->get_op_ptr()->is_parameter() && !gnode->get_op_ptr()->is_output() &&
                    !gnode->is_constant())
                {
                    auto emitted_kernels =
                        (*gnode)["Kernel_Selection_Result"]
                            .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                    auto emitter_iter =
                        find_if(emitted_kernels.begin(),
                                emitted_kernels.end(),
                                [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                    return (i.first == DeviceType::CUDA_GPU ||
                                            i.first == DeviceType::ROCM_GPU);
                                });

                    if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr ||
                        emitter_iter->second->get_or_emit_source() == nullptr)
                    {
                        CHECK_FAIL() << "Kernel should be emitted before this pass:"
                                     << gnode->get_name();
                    }
                    auto kernel = emitter_iter->second;
                    for (auto& in : kernel->m_context->input_names)
                    {
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }
                    if (const_inputs)
                    {
                        for (auto& out : kernel->m_context->output_names)
                        {
                            constant_vals.insert(out);
                        }
                        async_info.execution_stream = async_manager->set_stream(0, "default");
                    }
                }
                else
                {
                    for (size_t i = 0; i < gnode->get_input_size(); ++i)
                    {
                        auto in = gnode->get_input_tensor_ptr(i)->get_name();
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }
                    if (const_inputs)
                    {
                        for (size_t i = 0; i < gnode->get_output_size(); ++i)
                        {
                            auto out = gnode->get_output_tensor_ptr(i)->get_name();
                            constant_vals.insert(out);
                        }
                        async_info.execution_stream = async_manager->set_stream(0, "default");
                    }
                }
            }
        }
    }
    LOG(INFO) << "assign stream info-------------------------------";
}

void AssignAsyncInfoPass::assign_event_info(AsyncManager* async_manager,
                                            std::shared_ptr<Graph>& graph)
{
    for (auto gnode : graph->get_nodes())
    {
        CHECK((*gnode)["Async_info"].is_valid());
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        auto stream = async_info.execution_stream;
        CHECK_NOT_NULLPTR(stream);
        for (auto& edge : gnode->get_in_edges())
        {
            auto input_gnode = edge->get_src();
            // constant ops are in xxx_init() of generated code,
            // so there is no need to add event.
            if (input_gnode->get_op_ptr()->is_constant())
            {
                continue;
            }
            auto& input_async_info = (*input_gnode)["Async_info"].as<AsyncExecutionInfo>();
            auto input_stream = input_async_info.execution_stream;
            if (input_stream->get_device_name() == stream->get_device_name())
            {
                if (input_stream->get_stream_id() != stream->get_stream_id() &&
                    !input_gnode->get_op_ptr()->is_parameter())
                {
                    // Cuda streams perform implicite sychronization with default(0) stream,
                    // so there is no need to add event emplicitely.
                    if ((stream->get_device_type() == CUDA_GPU ||
                         stream->get_device_type() == ROCM_GPU) &&
                        (stream->is_default_stream() || input_stream->is_default_stream()))
                    {
                        continue;
                    }
                    if (input_async_info.record_event == nullptr)
                    {
                        input_async_info.record_event =
                            async_manager->set_event(input_stream, input_gnode->get_op_ptr());
                    }
                    async_info.wait_events.push_back(input_async_info.record_event);
                }
            }
            // todo: support cross-device events
            else
            {
                throw nnfusion::errors::NotSupported("Cross-device event is not supported.");
            }
        }
    }
    LOG(INFO) << "assign event info-------------------------------";
}