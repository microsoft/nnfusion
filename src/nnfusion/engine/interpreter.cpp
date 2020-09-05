// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "interpreter.hpp"
#include "nnfusion/engine/pass/codegen/cpu_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/cuda_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/rocm_codegen_pass.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"

#include <strings.h>
#include "nnfusion/common/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/pass/codegen/cuda_codegen_pass.hpp"
#include "pass/tensor/inplace_tensor_analysis.hpp"
#include "pass/tensor/liveness_analysis.hpp"
#include "pass/tensor/tensor_device_dispatcher.hpp"
#include "pass/tensor/tensor_memory_layout.hpp"
using namespace nnfusion::pass;

DECLARE_string(fdefault_device);
DEFINE_bool(fcuda_kernels_as_files, false, "Saving cuda kernels as standalone source code files.");
DEFINE_int64(fcuda_kernels_files_number,
             -1,
             "Saving cuda kernels into how many source code files.");

DEFINE_bool(fkernels_as_files, false, "Saving kernels as standalone source code files.");
DEFINE_int64(fkernels_files_number, -1, "Saving kernels into how many source code files.");
DECLARE_string(fcuda_init_stream);
DECLARE_string(fstream_assign_policy);
DEFINE_bool(ftraining_mode, false, "Turn on training mode.");
DEFINE_bool(fextern_result_memory, false, "Model result tensor memory is managed externally.");
DEFINE_int32(fwarmup_step, 5, "Warm up step.");
DEFINE_int32(frun_step, 100, "Run step.");

Interpreter::Interpreter()
    : m_trans_ctx(new InterpreterContext())
    , m_passes(new vector<shared_ptr<IInterpreterPass>>())
{
    // Todo: find another way
    auto dev_name = FLAGS_fdefault_device.c_str();
    NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

    // To be compatible with former cli
    //Todo(wenxh): Remove this;
    FLAGS_fkernels_as_files = FLAGS_fkernels_as_files || FLAGS_fcuda_kernels_as_files;
    FLAGS_fkernels_files_number =
        max(FLAGS_fkernels_files_number, FLAGS_fcuda_kernels_files_number);

    // kernel selection
    // m_passes->push_back(make_shared<DefaultDeviceDispatcher>());
    // m_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    // m_passes->push_back(make_shared<DefaultKernelSelector>());
    m_passes->push_back(make_shared<TensorDeviceDispatcher>());
    m_passes->push_back(make_shared<TensorLivenessAnalysis>());
    m_passes->push_back(make_shared<InplaceTensorAnalysis>());
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

    switch (default_device)
    {
    case CUDA_GPU: m_passes->push_back(make_shared<CudaCodegenPass>()); break;
    case GENERIC_CPU: m_passes->push_back(make_shared<CpuCodegenPass>()); break;

    case ROCM_GPU:
        FLAGS_fcuda_kernels_as_files = false;
        m_passes->push_back(make_shared<RocmCodegenPass>());
        break;

    default: m_passes->push_back(make_shared<CudaCodegenPass>()); break;
    }
}

Interpreter::Interpreter(shared_ptr<vector<shared_ptr<IInterpreterPass>>> passes,
                         shared_ptr<InterpreterContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

bool Interpreter::translate(TranslationUnit::Pointer tu)
{
    NNFUSION_CHECK_NOT_NULLPTR(m_passes);
    return IInterpreterPass::run_passes(*m_passes, m_trans_ctx, tu);
}

TranslationUnitMap& Interpreter::translate(shared_ptr<graph::Graph> graph)
{
    auto& _tus = m_trans_ctx->m_tus;
    if (_tus.find(graph) != _tus.end() && _tus[graph]->m_is_translated)
        return _tus;
    // run graph passes
    std::vector<shared_ptr<graph::Graph>> graph_vec{graph};
    nnfusion::pass::graph::GraphPass graph_passes;
    NNFUSION_CHECK(graph_passes.run(graph_vec) == true);

    // Iterator through all nodes
    static interpreter::ExtractGraphSignature extract_global;

    // Deal with translation unit's program
    for (auto cur_graph : graph_vec)
    {
        m_trans_ctx->m_graphs.insert(cur_graph);

        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        _tus.emplace(cur_graph, _tu);
        NNFUSION_LOG(INFO) << "Translating graph:\t" << cur_graph->get_name();

        _tu->program = nnfusion::ir::Program::create_single_basic_block_program();
        _tu->m_graph = cur_graph;
        auto bb_main = _tu->program.get_entry();

        // extract output_names/constants/arg/out for _tu, m_variable_name_map for m_trans_ctx
        NNFUSION_CHECK(extract_global.run(m_trans_ctx, _tu))
            << "Error when extract global graph info.";

        // Translate the Node
        nnfusion::graph::GNodeVector node_vec;
        if (FLAGS_fstream_assign_policy == "kernel_prof_based")
            node_vec = cur_graph->get_bfs_ordered_ops();
        else
            node_vec = cur_graph->get_ordered_ops();
        for (auto gnode : node_vec)
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(gnode);
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
                ir->setGNode(gnode);

                // Tag example
                {
                    auto& INS = *ir;
                    INS["DEBUG"] = 1;
                    auto res = INS["DEBUG"].as<int>();
                }

                // move all tags on the node to the intruction
                {
                    ir->copy_tags_from(*gnode);
                }

                ir->setName(gnode->get_name());
                bb_main->push_back(ir);

                // add memcpy ir and async info if the gnode and its output gnodes are in diffrent devices
                add_memcpy_ir(cur_graph, gnode, bb_main);
            }
        }
        if (translate(_tu))
            _tu->m_is_translated = true;
    }
    return m_trans_ctx->m_tus;
}

void Interpreter::add_memcpy_ir(shared_ptr<graph::Graph> graph,
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
