// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reversed_dfs_visitor.hpp"
#include "nnfusion/common/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"

using namespace nnfusion;

DECLARE_string(fcuda_init_stream);
DECLARE_string(fstream_assign_policy);

void add_memcpy_ir(shared_ptr<graph::Graph> graph,
                   shared_ptr<nnfusion::graph::GNode> gnode,
                   nnfusion::ir::BasicBlock::Pointer bb_main)
{
    auto CUDA_async_manager =
        nnfusion::async::AsyncManagerFactory::get_device_stream_async_manager(graph, CUDA_GPU);
    auto CPU_async_manager =
        nnfusion::async::AsyncManagerFactory::get_host_async_manager(graph, GENERIC_CPU);
    std::unordered_map<string, nnfusion::ir::Instruction::Pointer> dev_ir;
    auto n_device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
    auto n_device_id = (*gnode)["DeviceID"].as<int>();
    for (auto& out_edge : gnode->get_out_edges())
    {
        if (!out_edge->is_control_edge())
        {
            auto out_gnode = out_edge->get_dst();

            auto out_device_type = (*out_gnode)["DeviceType"].as<NNFusion_DeviceType>();
            auto out_device_id = (*out_gnode)["DeviceID"].as<int>();
            std::string out_dev_name =
                get_device_str(out_device_type) + "_" + to_string(out_device_id);
            if (n_device_type != out_device_type || n_device_id != out_device_id)
            {
                nnfusion::ir::Instruction::Pointer memcpy_ir;
                if (dev_ir.find(out_dev_name) == dev_ir.end())
                {
                    auto& async_info =
                        (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                    auto thread = async_info.execution_thread;
                    auto stream = async_info.execution_stream;
                    memcpy_ir = std::make_shared<nnfusion::ir::Instruction>();
                    memcpy_ir->setName("Memcpy");
                    if (n_device_type == out_device_type)
                    {
                        // set device id and device type
                        (*memcpy_ir)["DeviceType"] = n_device_type;
                        (*memcpy_ir)["DeviceID"] = n_device_id;
                    }
                    else if (n_device_type == GENERIC_CPU)
                    {
                        // set device id and device type
                        (*memcpy_ir)["DeviceType"] = out_device_type;
                        (*memcpy_ir)["DeviceID"] = out_device_id;
                    }
                    else if (out_device_type == GENERIC_CPU)
                    {
                        // set device id and device type
                        (*memcpy_ir)["DeviceType"] = n_device_type;
                        (*memcpy_ir)["DeviceID"] = n_device_id;
                    }
                    auto idx = out_edge->get_dst_input();
                    auto tensor = out_gnode->get_input_tensor_ptr(idx);
                    auto element_type = tensor->get_element_type();
                    auto pshape = tensor->get_partial_shape();
                    auto name = tensor->get_name();
                    std::string new_name = name + "_" + get_device_str(out_device_type) +
                                           std::to_string(out_device_id);
                    // create new tensor
                    std::shared_ptr<descriptor::Tensor> new_tensor =
                        std::make_shared<descriptor::Tensor>(element_type, pshape, new_name);
                    // set tensor layout
                    auto layout = std::make_shared<nnfusion::descriptor::layout::DenseTensorLayout>(
                        *new_tensor);
                    new_tensor->set_tensor_layout(layout);

                    auto& inputs = memcpy_ir->get_inputs();
                    NNFUSION_CHECK(inputs.empty());
                    inputs.push_back(tensor);

                    auto& outputs = memcpy_ir->get_outputs();
                    NNFUSION_CHECK(outputs.empty());
                    outputs.push_back(new_tensor);

                    (*memcpy_ir)["Async_info"] = nnfusion::async::AsyncExecutionInfo();
                    auto& memcpy_async_info =
                        (*memcpy_ir)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                    // set thread and stream
                    if (gnode->get_op_ptr()->is_tensor_op())
                    {
                        new_tensor->set_persistent();
                    }
                    if (gnode->is_constant() || gnode->is_variable())
                    {
                        (*memcpy_ir)["Memcpy_Constant_or_Variable"] = true;
                    }
                    if ((*gnode)["rt_const_folding"].is_valid_as<bool>())
                    {
                        (*memcpy_ir)["rt_const_folding"] = true;
                    }
                    if (gnode->is_constant() || gnode->is_variable() ||
                        (*gnode)["rt_const_folding"].is_valid_as<bool>())
                    {
                        NNFUSION_CHECK(thread->is_default_stream());
                        //NNFUSION_CHECK(stream->is_default_stream());
                        memcpy_async_info.execution_thread = thread;
                        if (stream)
                            memcpy_async_info.execution_stream = stream;
                        else
                            memcpy_async_info.execution_stream = CUDA_async_manager->set_stream(
                                (*memcpy_ir)["DeviceID"].as<int>(), FLAGS_fcuda_init_stream);
                    }
                    else
                    {
                        if (gnode->is_parameter())
                            memcpy_async_info.execution_thread =
                                CPU_async_manager->set_stream(0, "memcpy");
                        else
                            memcpy_async_info.execution_thread = thread;
                        // use a new different stream.
                        memcpy_async_info.execution_stream = CUDA_async_manager->set_stream(
                            (*memcpy_ir)["DeviceID"].as<int>(), "memcpy_" + new_name);
                    }
                    if (gnode->is_constant() || gnode->is_variable() ||
                        (*gnode)["rt_const_folding"].is_valid_as<bool>())
                    {
                        // constant, variable and rt_const_folding ops are in xxx_init(),
                        // so thre is no need to add event or barrier.
                    }
                    else
                    {
                        if (memcpy_async_info.execution_thread != thread &&
                            async_info.notify_barrier != nullptr)
                        {
                            memcpy_async_info.wait_barriers.push_back(async_info.notify_barrier);
                        }
                        if (stream && memcpy_async_info.execution_stream != stream &&
                            async_info.record_event != nullptr)
                        {
                            memcpy_async_info.wait_events.push_back(async_info.record_event);
                        }
                    }
                    bb_main->push_back(memcpy_ir);
                    dev_ir[out_dev_name] = memcpy_ir;
                }
                else
                {
                    memcpy_ir = dev_ir[out_dev_name];
                }

                auto& inputs = memcpy_ir->get_inputs();
                NNFUSION_CHECK(inputs.size() == 1);
                auto tensor = inputs[0];
                auto name = tensor->get_name();

                auto& outputs = memcpy_ir->get_outputs();
                NNFUSION_CHECK(outputs.size() == 1);
                auto new_tensor = outputs[0];
                auto new_name = new_tensor->get_name();

                if (!out_gnode->get_op_ptr()->is_tensor_op())
                {
                    auto out_kernel = (*out_gnode)["Kernel_Selection_Result"]
                                          .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>()
                                          .second;
                    NNFUSION_CHECK_NOT_NULLPTR(out_kernel);
                    auto& out_kernel_inputs = out_kernel->m_context->inputs;
                    auto& out_kernel_input_names = out_kernel->m_context->input_names;
                    for (size_t i = 0; i < out_kernel_input_names.size(); ++i)
                    {
                        if (out_kernel_input_names[i] == name)
                        {
                            out_kernel_inputs.erase(out_kernel_inputs.begin() + i);
                            out_kernel_inputs.insert(out_kernel_inputs.begin() + i, new_tensor);
                            out_kernel_input_names.erase(out_kernel_input_names.begin() + i);
                            out_kernel_input_names.insert(out_kernel_input_names.begin() + i,
                                                          new_name);
                            break;
                        }
                    }
                }
                else
                {
                    // tensor op nodes have no inputs,
                    // and output nodes kernel is simply a reference operation,
                    // so there should be no memcpy between these nodes and their input nodes.
                }

                // add waiting event and barrier to the out gnodes
                // constant, variable, and rt_const_folding ops memcpy ir are in xxx_init(),
                // so there is no need to add event or barrier.
                if ((*memcpy_ir)["Memcpy_Constant_or_Variable"].is_valid_as<bool>() ||
                    (*memcpy_ir)["rt_const_folding"].is_valid_as<bool>())
                    continue;
                auto& memcpy_async_info =
                    (*memcpy_ir)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                auto& out_async_info =
                    (*out_gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                if (memcpy_async_info.execution_thread != out_async_info.execution_thread)
                {
                    if (!memcpy_async_info.notify_barrier)
                    {
                        memcpy_async_info.notify_barrier = CPU_async_manager->set_event(
                            memcpy_async_info.execution_thread, "memcpy_" + new_name);
                    }
                    out_async_info.wait_barriers.push_back(memcpy_async_info.notify_barrier);
                }
                else
                {
                    auto memcpy_dev = (*memcpy_ir)["DeviceType"].as<NNFusion_DeviceType>();
                    auto out_dev = (*out_gnode)["DeviceType"].as<NNFusion_DeviceType>();
                    if (memcpy_dev != out_dev)
                        memcpy_async_info.sync_stream = true;
                }

                if (out_async_info.execution_stream &&
                    memcpy_async_info.execution_stream != out_async_info.execution_stream)
                {
                    if (!memcpy_async_info.record_event)
                    {
                        memcpy_async_info.record_event = CUDA_async_manager->set_event(
                            memcpy_async_info.execution_stream, "memcpy_" + new_name);
                    }
                    out_async_info.wait_events.push_back(memcpy_async_info.record_event);
                }
            }
        }
    }
}

nnfusion::ir::Program::Pointer ReversedDFSVisitor::run_on_graph(shared_ptr<graph::Graph> graph,
                                                                EngineContext::Pointer context)
{
    NNFUSION_LOG(INFO) << "Translating graph:\t" << graph->get_name();

    auto program =
        make_shared<ir::Program>(nnfusion::ir::Program::create_single_basic_block_program());
    auto bb_main = program->get_entry();

    // Translate the Node
    // Currently:
    // * Translate each gnode into an instruction;
    // * Store all instruction inside one basicblock since we don't have
    //   control-flow by now.
    nnfusion::graph::GNodeVector node_vec;
    if (FLAGS_fstream_assign_policy == "kernel_prof_based")
        node_vec = graph->get_bfs_ordered_ops();
    else
        node_vec = graph->get_ordered_ops();

    for (auto gnode : node_vec)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
        ir->setGNode(gnode);
        ir->copy_tags_from(*gnode);
        ir->setName(gnode->get_name());
        bb_main->push_back(ir);
        add_memcpy_ir(graph, gnode, bb_main);
    }

    return program;
}