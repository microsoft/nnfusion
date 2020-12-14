// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_codegen_pass.hpp"
#include "codegen_langunit.hpp"
#include "codegenerator_helper.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

DEFINE_bool(fcodegen_debug, false, "Add debug functions in Codegen-ed project.");
DECLARE_string(fdefault_device);
DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(frt_const_folding);
DECLARE_string(fcuda_init_stream);
DECLARE_bool(fextern_result_memory);
DECLARE_int32(fwarmup_step);
DECLARE_int32(frun_step);

void CudaCodegenPass::set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu)
{
    this->device_async_manager =
        AsyncManagerFactory::get_device_stream_async_manager(tu->m_graph, CUDA_GPU);
    this->host_async_manager =
        AsyncManagerFactory::get_host_async_manager(tu->m_graph, GENERIC_CPU);

    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            auto kernel = ins->getKernel();
            if (!kernel || !kernel->get_or_emit_source())
                continue;
            for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
            {
                global_required.insert(it.second->symbol);
            }
        }
    }

    superscaler_enable = global_required.count("header::super_scaler") > 0;
    return;
}

void CudaCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    set_global_member(ctx, tu);

    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.cu";
    auto& copy_templates = projgen->lup_codegen->copy_templates;
    if (superscaler_enable)
    {
        copy_templates.emplace_back("super_scaler/super_scaler.h", "./super_scaler.h");
        NNFUSION_LOG(NNFUSION_WARNING) << "libsuper_scaler.so should be copied from "
                                          "(build)/src/tools/nnfusion/templates/super_scaler/";
        copy_templates.emplace_back("super_scaler/libsuper_scaler.so", "./libsuper_scaler.so");
    }

    copy_templates.emplace_back("image_tests/image_test.cpp", "./image_tests/image_test.cpp");
    copy_templates.emplace_back("image_tests/CMakeLists_cuda.txt", "./image_tests/CMakeLists.txt");
    //copy folder
    auto& copy_folder = projgen->lup_codegen->copy_folder;
    char exe_path[PATH_MAX];
    size_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
    const char* path;
    if (count != -1)
    {
        path = dirname(exe_path);
    }
    else
    {
        throw nnfusion::errors::RuntimeError("Failed to get the directory of executable file.\n");
    }

    if (host_async_manager && host_async_manager->num_non_default_stream() > 0)
    {
        std::string eigen_path = std::string(path) + std::string("/eigen");
        copy_folder.push_back(eigen_path);
        std::string threadpool_path = std::string(path) + std::string("/threadpool");
        copy_folder.push_back(threadpool_path);
    }

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        lu_init_begin << "\nextern \"C\" void cuda_init()\n{\n";
        lu_init_begin << "CUDA_SAFE_CALL(cudaDeviceReset());\n";
    }

    auto& lu_init_end = *(projgen->lup_init->end);
    {
        lu_init_end << "}\n\n";
    }

    auto& lu_exec_begin = *(projgen->lup_exec->begin);
    {
        std::string params = get_kernel_entry_paras(tu);
        lu_exec_begin << "\nextern \"C\" int kernel_entry(" << params << ")\n{\n";
    }

    auto& lu_exec_init = *(projgen->lup_exec->begin);
    {
        auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();
        lu_exec_init << "// kernel_entry_init\n";
        // emit memset
        for (const auto& allocator : allocator_list)
        {
            if (allocator.first.find("memset") != std::string::npos)
            {
                lu_exec_init << allocator.second->emit_memory_set(0)->get_code();
            }
        }
    }

    auto& lu_exec_end = *(projgen->lup_exec->end);
    {
        lu_exec_end << "return 0;\n";
        lu_exec_end << "}\n\n";
    }

    auto& lu_exit_begin = *(projgen->lup_exit->begin);
    {
        lu_exit_begin << "\nextern \"C\" void cuda_free()\n{\n";
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        lu_exit_end << "}\n\n";
    }

    // add component
    create_graph_config(ctx, tu);
    create_header_file(ctx, tu);
    create_main_file(ctx, tu);
    create_cmake_file(ctx, tu);

    // add requirement
    projgen->lup_codegen->require(header::assert);
    projgen->lup_codegen->require(header::stdexcept);
    projgen->lup_codegen->require(header::sstream);
    projgen->lup_codegen->require(header::cuda);
    projgen->lup_codegen->require(header::cublas);
    projgen->lup_codegen->require(header::cudnn);
    projgen->lup_codegen->require(macro::CUDA_SAFE_CALL);
    projgen->lup_codegen->require(macro::CUDNN_SAFE_CALL);
    projgen->lup_codegen->require(macro::CUBLAS_SAFE_CALL);
    projgen->lup_codegen->require(macro::HALF_MAX);

    return;
}

bool CudaCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

    // collect code
    auto pairs = collect_ins(ctx, tu);
    for (size_t i = 0; i < pairs.size(); i++)
    {
        auto& it = pairs[i];
        int pos = it.first.find(":");
        NNFUSION_CHECK(pos >= 0);
        std::string thread_name = it.first.substr(pos + 1);
        std::string main_block = it.first.substr(0, pos);

        auto lup_func_calls = get_kernel_func_calls(it.first + "_func_calls", nullptr);

        auto thread_call_paras_args_pair = get_paras_and_args(it.second);
        auto thread_call_paras = thread_call_paras_args_pair.first;
        auto thread_call_args = thread_call_paras_args_pair.second;
        if (!thread_call_args.empty())
            thread_call_args = ", " + thread_call_args;

        bool func_call_only = (main_block == "init");

        for (auto ins : it.second)
        {
            auto kernel = ins->getKernel();
            auto gnode = ins->getGNode();
            auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string body_str = fu->body_unit->get_code();
            string func_name = fu->name_unit->get_code();
            // conv kernels in the the stream shares the same workspace_ptr
            if (gnode->get_op_type() == "Convolution" && async_info.execution_stream)
            {
                std::string s_workspace =
                    "workspace_ptr_" + to_string(async_info.execution_stream->get_stream_id());
                int pos = body_str.find("workspace_ptr");
                while (pos >= 0)
                {
                    body_str.replace(pos, 13, s_workspace);
                    pos = body_str.find("workspace_ptr", pos + s_workspace.size());
                }
            }
            body_str = fu->signature_unit->get_code() + body_str;

            if (!body_str.empty())
            {
                if (kernel->is_static_function() ||
                    kernel_func_defs.find(body_str) == kernel_func_defs.end())
                {
                    if (!kernel->is_eliminative())
                    {
                        auto kernel_func_def = codegenerator::FunctionFile::convert_from(kernel);

                        if (!kernel->is_static_function())
                            kernel_func_defs[body_str] = std::make_pair(func_name, kernel_func_def);
                    }
                }
                else
                {
                    func_name = kernel_func_defs[body_str].first;
                }

                if (kernel_func_defs.find(body_str) != kernel_func_defs.end())
                {
                    lup_func_calls->require(kernel_func_defs[body_str].second);
                    if (FLAGS_fkernels_as_files &&
                        kernel_func_defs[body_str].second->extern_decl_unit != nullptr)
                        lup_func_calls->require(
                            kernel_func_defs[body_str].second->extern_decl_unit);
                }
            }

            std::string call_str = fu->get_specialized_funciton_call(func_name);
            int pos_right = call_str.find(">>>(");
            if (pos_right >= 0)
            {
#ifdef __USING_HOST_CALL_FORMAT___
                // Turn to Host Call Format in kernel_entry()
                int pos_left = call_str.find("<<<");
                NNFUSION_CHECK(pos_left >= 0);
                call_str = call_str.substr(0, pos_left) + "_Call(" +
                           call_str.substr(pos_left + sizeof("<<<") - 1);

                pos_right = call_str.find(">>>(");
                NNFUSION_CHECK(pos_right >= 0);
                call_str = call_str.substr(0, pos_right) + ", " +
                           call_str.substr(pos_right + sizeof(">>>(") - 1);
#endif
            }
            LanguageUnit_p kernel_func_call = func_call_codegen(ins, func_call_only, call_str);
            lup_func_calls->unit_vec.push_back(kernel_func_call);
        }

        if (thread_name != "default_thread")
        {
            LanguageUnit_p new_caller =
                std::make_shared<LanguageUnit>(lup_func_calls->symbol + "_new_caller");
            auto& lu_new_caller = *new_caller;
            {
                // add function call to kernel entry
                std::string std_thread_func_name = std::string("thread_func") + std::to_string(i);
                std::string thread_call_str =
                    std::string("(") + thread_name + thread_call_args + std::string(");\n");
                std::string std_thread_func_call = std::string("auto ") + std_thread_func_name +
                                                   std::string(" = std::bind") + thread_call_str;
                lu_new_caller << std_thread_func_call;
                std::string t_threadpool_call = std::string("schedule_thread_pool->Schedule(");
                t_threadpool_call += (std_thread_func_name + std::string(");\n"));
                lu_new_caller << t_threadpool_call;
            }

            LanguageUnit_p begin =
                std::make_shared<LanguageUnit>(lup_func_calls->symbol + "_new_caller_block_begin");
            auto& lu_begin = *begin;
            {
                lu_begin << "extern \"C\" void " << thread_name << "(";
                lu_begin << thread_call_paras << ")\n{\n";
            }

            LanguageUnit_p end =
                std::make_shared<LanguageUnit>(lup_func_calls->symbol + "_new_caller_block_end");
            auto& lu_end = *end;
            {
                lu_end << "default_barrier.Notify();\n}\n\n";
            }

            new_caller = lup_func_calls->wrap(new_caller, begin, end);
            if (main_block == "init")
                projgen->lup_init->unit_vec.push_back(new_caller);
            else if (main_block == "exec")
                projgen->lup_exec->unit_vec.push_back(new_caller);
            else
                NNFUSION_CHECK_FAIL() << "Unrecognized main_block";
        }
        else
        {
            if (main_block == "init")
                projgen->lup_init->unit_vec.push_back(lup_func_calls);
            else if (main_block == "exec")
                projgen->lup_exec->unit_vec.push_back(lup_func_calls);
            else
                NNFUSION_CHECK_FAIL() << "Unrecognized main_block";
        }
    }

    if (FLAGS_fkernels_as_files)
        separate_func_defs_files(FLAGS_fkernels_files_number, m_codegen_folder + "kernels/");

    return true;
}

std::vector<std::pair<string, vector<nnfusion::ir::Instruction::Pointer>>>
    CudaCodegenPass::collect_ins(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    unordered_map<string, vector<nnfusion::ir::Instruction::Pointer>> ins_vec_map;

    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            auto kernel = ins->getKernel();
            auto gnode = ins->getGNode();

            if (gnode && gnode->is_parameter())
                continue;
            // if (kernel && kernel->is_eliminative())
            //     continue;
            if (kernel && kernel->get_or_emit_source())
            {
                // do nothing
            }
            else
            {
                auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                    "AnyOP", device_type(), element::f32);
                NNFUSION_CHECK(kernel_reg != nullptr) << "AnyOp Kernel not found, op="
                                                      << ins->getGNode()->get_op_type();
                shared_ptr<KernelContext> ctx(new KernelContext(ins->getGNode()));
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();
                ins->setKernel(kernel);
            }

            auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
            int device_id = (*ins)["DeviceID"].as<int>();
            auto thread = async_info.execution_thread;
            auto thread_name = thread->get_name();
            std::string key;
            if (gnode->is_constant() || gnode->is_variable() ||
                (FLAGS_frt_const_folding && (*ins)["rt_const_folding"].is_valid_as<bool>()))
            {
                key = "init:" + thread_name;
            }
            else
            {
                key = "exec:" + thread_name;
            }
            ins_vec_map[key].push_back(ins);
        }
    }

    std::vector<std::pair<string, vector<nnfusion::ir::Instruction::Pointer>>> pairs;
    for (auto itr = ins_vec_map.begin(); itr != ins_vec_map.end(); ++itr)
        pairs.push_back(*itr);
    //if superscaler_enable, we preserve the thread call order
    if (superscaler_enable)
    {
        sort(pairs.begin(),
             pairs.end(),
             [](std::pair<string, vector<nnfusion::ir::Instruction::Pointer>>& a,
                std::pair<string, vector<nnfusion::ir::Instruction::Pointer>>& b) {
                 if (a.first.find("default_") != string::npos)
                     return false;

                 if (b.first.find("default") != string::npos)
                     return false;

                 int pos_a = a.first.find("async_");
                 int pos_b = b.first.find("async_");
                 if (pos_a >= 0 && pos_b >= 0)
                 {
                     string delimiter("_");
                     std::string d1 = a.first.substr(a.first.find(delimiter) + 1);
                     d1 = d1.substr(0, d1.find(delimiter));
                     std::string d2 = b.first.substr(b.first.find(delimiter) + 1);
                     d2 = d2.substr(0, d2.find(delimiter));
                     return std::stoi(d1) < std::stoi(d2);
                 }

                 return a.first > b.first;
             });
    }

    return pairs;
}

std::string CudaCodegenPass::get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu)
{
    unordered_set<string> allocated;
    vector<string> params;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        ss << type << "* " << tv->get_name();
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }

    for (int i = 0; i < tu->out.size(); i++)
    {
        auto tv = tu->out[i];
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        if (FLAGS_fextern_result_memory)
            ss << type << "* " << tv->get_name();
        else
            ss << type << "** " << tv->get_name();
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }
    return join(params, ", ");
}

std::pair<std::string, std::string>
    CudaCodegenPass::get_paras_and_args(std::vector<nnfusion::ir::Instruction::Pointer>& ir_vec)
{
    std::pair<std::string, std::string> paras_and_args;
    vector<string> params;
    vector<string> args;
    unordered_set<string> allocated;
    for (auto ins : ir_vec)
    {
        auto kernel = ins->getKernel();
        if (kernel && kernel->m_context)
        {
            for (auto input : kernel->m_context->inputs)
            {
                auto name = input->get_name();
                if (allocated.find(name) == allocated.end() &&
                    name.compare(0, 10, "Parameter_") == 0)
                {
                    string type = input->get_element_type().c_type_string();
                    stringstream ss;
                    ss << type << "* " << name;
                    allocated.insert(name);
                    params.push_back(ss.str());
                    args.push_back(name);
                }
            }
            if (kernel->m_context->gnode && kernel->m_context->gnode->get_op_ptr()->is_output())
            {
                for (auto output : kernel->m_context->outputs)
                {
                    auto name = output->get_name();
                    if (allocated.find(name) == allocated.end())
                    {
                        string type = output->get_element_type().c_type_string();
                        stringstream ss;
                        if (FLAGS_fextern_result_memory)
                            ss << type << "* " << name;
                        else
                            ss << type << "** " << name;
                        allocated.insert(name);
                        params.push_back(ss.str());
                        args.push_back(name);
                    }
                }
            }
        }
    }
    paras_and_args.first = join(params, ", ");
    paras_and_args.second = join(args, ", ");
    return paras_and_args;
}

nnfusion::LanguageUnit_p CudaCodegenPass::func_call_codegen(nnfusion::ir::Instruction::Pointer ins,
                                                            bool func_call_only,
                                                            const std::string& func_call)
{
    std::string symbol;
    auto kernel = ins->getKernel();
    if (kernel)
    {
        auto fu = kernel->get_or_emit_source();
        symbol = fu->call_unit->get_symbol();
    }
    else
    {
        symbol = ins->name(); // todo: make symbol unique
    }

    auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
    LanguageUnit_p _lu(new LanguageUnit(symbol));
    auto& lu = *_lu;

    auto gnode = ins->getGNode();
    std::string node_name = gnode ? gnode->get_name() : ins->name();

    lu << " // name=" << node_name << "\n";
    if (!func_call_only)
    {
        if (!async_info.wait_barriers.empty())
        {
            for (auto barrier : async_info.wait_barriers)
            {
                lu << host_async_manager->emit_event_wait(async_info.execution_thread, barrier)
                          ->get_code();
            }
        }
        if (!async_info.wait_events.empty())
        {
            for (auto event : async_info.wait_events)
            {
                lu << device_async_manager->emit_event_wait(async_info.execution_stream, event)
                          ->get_code();
            }
        }
    }

    if (ins->name() == "Memcpy")
    {
        //string stream_name = async_info.execution_stream->get_name();
        auto& inputs = ins->get_inputs();
        NNFUSION_CHECK(inputs.size() == 1);
        auto src_tensor = inputs[0];
        auto& outputs = ins->get_outputs();
        NNFUSION_CHECK(outputs.size() == 1);
        auto dst_tensor = outputs[0];
        std::string memcpykind;
        auto dst_dev = dst_tensor->get_device_type();
        auto src_dev = src_tensor->get_device_type();
        if (dst_dev == src_dev)
        {
            NNFUSION_CHECK(dst_dev == CUDA_GPU || dst_dev == ROCM_GPU);
            memcpykind = ", cudaMemcpyDeviceToDevice, ";
        }
        else if (dst_dev == GENERIC_CPU)
        {
            NNFUSION_CHECK(src_dev == CUDA_GPU || src_dev == ROCM_GPU);
            memcpykind = ", cudaMemcpyDeviceToHost, ";
        }
        else if (src_dev == GENERIC_CPU)
        {
            NNFUSION_CHECK(dst_dev == CUDA_GPU || dst_dev == ROCM_GPU);
            memcpykind = ", cudaMemcpyHostToDevice, ";
        }
        else
        {
            nnfusion::errors::NotSupported("Unsupported memcpy kind.");
        }
        string stream_name = async_info.execution_stream->get_name();
        lu << "cudaMemcpyAsync(" << dst_tensor->get_name() << ", " << src_tensor->get_name() << ", "
           << dst_tensor->size() << memcpykind << stream_name << ");\n";
    }
    else
    {
        if (ins->getKernel()->is_eliminative())
        {
            lu << "// eliminated: " << func_call;
        }
        else
        {
            lu << func_call;
        }
    }

    if (!func_call_only)
    {
        if (async_info.record_event != nullptr)
        {
            lu << device_async_manager->emit_event_record(async_info.record_event)->get_code();
        }
        if (async_info.notify_barrier != nullptr)
        {
            lu << host_async_manager->emit_event_record(async_info.notify_barrier)->get_code();
        }
    }

    if (ins->name() == "Memcpy" && async_info.sync_stream == true)
    {
        lu << "CUDA_SAFE_CALL(cudaStreamSynchronize(" << async_info.execution_stream->get_name()
           << "));\n";
    }

    if (FLAGS_fcodegen_debug && gnode && kernel && !gnode->get_op_ptr()->is_output())
    {
        for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
        {
            if (kernel->m_context->outputs[i]->get_element_type().c_type_string() != "float")
                continue;
            auto out_name = kernel->m_context->output_names[i];
            lu << "Debug(\"" << node_name << ", " << out_name << "\", " << out_name << ", \""
               << join(kernel->m_context->input_names) << "\", "
               << kernel->m_context->outputs[i]->size(false) << ");\n";
        }
        lu.require(codegen::helper::debug);
    }

    return _lu;
}

bool CudaCodegenPass::collect_stream(std::shared_ptr<InterpreterContext> ctx,
                                     std::shared_ptr<TranslationUnit> tu)
{
    //stream
    NNFUSION_CHECK_NOT_NULLPTR(device_async_manager);
    if (device_async_manager && device_async_manager->num_stream() > 0)
    {
        auto stream_decl = device_async_manager->emit_stream_decl();
        auto stream_init = device_async_manager->emit_stream_init();
        auto stream_destroy = device_async_manager->emit_stream_destroy();

        stream_init->require(stream_decl);
        add_init_and_exit_pair(stream_init, stream_destroy);
    }
    //event
    if (device_async_manager && device_async_manager->num_event() > 0)
    {
        auto event_decl = device_async_manager->emit_event_decl();
        auto event_init = device_async_manager->emit_event_init();
        auto event_destroy = device_async_manager->emit_event_destroy();

        event_init->require(event_decl);
        add_init_and_exit_pair(event_init, event_destroy);
    }

    return true;
}

bool CudaCodegenPass::modify_codegen()
{
    if (global_required.count("declaration::num_SMs") > 0)
    {
        projgen->lup_init->unit_vec.push_back(
            std::make_shared<LanguageUnit>("num_SMs_init",
                                           "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                                           "cudaDevAttrMultiProcessorCount, 0));\n"));
        projgen->lup_init->require(declaration::num_SMs);
    }

    //dropout
    for (auto dep : global_required)
    {
        size_t position = dep.find("declaration::dropout_");
        if (position == dep.npos)
            continue;
        position = dep.find("dropout_");
        std::string dropout_name = dep.substr(position);
        auto dropout_pair = create_init_and_exit_pair<LanguageUnit, LanguageUnit>(
            dropout_name + "_init", dropout_name + "_free");
        auto& lu_dropout_init = *(dropout_pair.first);
        lu_dropout_init << dropout_name << "_init(cudnn_handle_0);\n";
        auto& lu_dropout_free = *(dropout_pair.second);
        lu_dropout_free << dropout_name << "_free();\n";
    }

    if (global_required.count("header::super_scaler"))
    {
        projgen->lup_exec->unit_vec.push_back(
            std::make_shared<LanguageUnit>("super_scaler_sync", "super_scaler_sync();\n"));
        auto super_scaler_pair = create_init_and_exit_pair<LanguageUnit, LanguageUnit>(
            "super_scaler_init", "super_scaler_final");
        auto& lu_ss_init = *(super_scaler_pair.first);
        lu_ss_init << "super_scaler_initialization();\n";
        auto& lu_ss_final = *(super_scaler_pair.second);
        lu_ss_final << "super_scaler_finalization();\n";
    }

    // multi-thread
    if (host_async_manager && host_async_manager->num_non_default_stream() > 0)
    {
        projgen->lup_codegen->require(header::threadpool);
        projgen->lup_codegen->require(declaration::schedule_thread_pool);
        projgen->lup_codegen->require(header::barrier);
        // auto thread_decl = host_async_manager->emit_stream_decl();
        // projgen->lup_codegen->require(thread_decl);

        LanguageUnit_p default_barrier_decl =
            std::make_shared<LanguageUnit>("declaration::default_barrier_decl");
        projgen->lup_codegen->require(default_barrier_decl);
        auto& lu_default_barrier_decl = *default_barrier_decl;
        {
            lu_default_barrier_decl << "nnfusion::cpu::Barrier default_barrier("
                                    << host_async_manager->num_non_default_stream() << ");\n";
        }

        auto& body = projgen->lup_exec->unit_vec;
        LanguageUnit_p default_barrier_reset =
            std::make_shared<LanguageUnit>("default_barrier_reset", "default_barrier.Reset();\n");
        body.insert(body.begin(), default_barrier_reset);
        body.insert(body.begin(), host_async_manager->emit_event_reset());
        LanguageUnit_p default_barrier_wait =
            std::make_shared<LanguageUnit>("default_barrier_wait", "default_barrier.Wait();\n");
        body.push_back(default_barrier_wait);

        auto schedule_thread_pool_pair = create_init_and_exit_pair<LanguageUnit, LanguageUnit>(
            "init_schedule_thread_pool", "del_schedule_thread_pool");
        auto lup_schedule_thread_pool_init = schedule_thread_pool_pair.first;
        auto lup_schedule_thread_pool_del = schedule_thread_pool_pair.second;

        auto& lu_schedule_thread_pool_init = *lup_schedule_thread_pool_init;
        {
            lu_schedule_thread_pool_init
                << "schedule_thread_pool = new concurrency::NumaAwareThreadPool();\n";
        }
        auto& lu_schedule_thread_pool_del = *lup_schedule_thread_pool_del;
        {
            lu_schedule_thread_pool_del << "delete schedule_thread_pool;\n";
        }
    }

    if (host_async_manager && host_async_manager->num_event() > 0)
    {
        auto barrier_decl = host_async_manager->emit_event_decl();
        projgen->lup_codegen->require(barrier_decl);
    }

    if (host_async_manager &&
        (host_async_manager->num_event() > 0 || host_async_manager->num_non_default_stream() > 0))
    {
        projgen->lup_codegen->require(barrier_header);
        barrier_header->write_to = barrier_header->symbol;
    }

    if (FLAGS_fcuda_init_stream != "default")
    {
        LanguageUnit_p device_sync = std::make_shared<LanguageUnit>(
            "cudaDeviceSynchronize", "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n");
        projgen->lup_init->unit_vec.push_back(device_sync);
    }

    return true;
}

void CudaCodegenPass::create_graph_config(std::shared_ptr<InterpreterContext> ctx,
                                          std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lu = make_shared<LanguageUnit>("GRAPH_CONFIG");
    LanguageUnit& lu_graph_config = *lu;
    lu_graph_config << "\n#ifndef __NNFUSION_GRAPH_CONFIG__\n";
    lu_graph_config << "#define __NNFUSION_GRAPH_CONFIG__\n";
    lu_graph_config << "#define NNFUSION_GRAPH_INPUT_NUM " << tu->arg.size() << "\n";
    lu_graph_config << "#define NNFUSION_GRAPH_OUTPUT_NUM " << tu->out.size() << "\n";
    for (int i = 0; i < tu->arg.size(); i++)
    {
        lu_graph_config << "#define NNFUSION_GRAPH_INPUT_DTYPE_" << i << " "
                        << tu->arg[i]->get_element_type().c_type_string() << "\n";
        lu_graph_config << "#define NNFUSION_GRAPH_INPUT_SHAPE_" << i << " {";
        auto& shape = tu->arg[i]->get_shape();
        for (int j = 0; j < shape.size(); ++j)
        {
            if (j > 0)
                lu_graph_config << ", ";
            lu_graph_config << shape[j];
        }
        lu_graph_config << "}\n";
    }
    for (int i = 0; i < tu->out.size(); i++)
    {
        lu_graph_config << "#define NNFUSION_GRAPH_OUTPUT_DTYPE_" << i << " "
                        << tu->out[i]->get_element_type().c_type_string() << "\n";
        lu_graph_config << "#define NNFUSION_GRAPH_OUTPUT_SHAPE_" << i << " {";
        auto& shape = tu->out[i]->get_shape();
        for (int j = 0; j < shape.size(); ++j)
        {
            if (j > 0)
                lu_graph_config << ", ";
            lu_graph_config << shape[j];
        }
        lu_graph_config << "}\n";
    }
    lu_graph_config << "#endif\n\n";

    projgen->lup_codegen->require(lu);
}

void CudaCodegenPass::create_header_file(std::shared_ptr<InterpreterContext> ctx,
                                         std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_header = std::make_shared<LanguageUnit>("codegen_header");
    projgen->lup_codegen->require(lup_header);
    lup_header->pwd = m_codegen_folder;
    lup_header->write_to = "nnfusion_rt.h";

    auto& lu_header = *lup_header;
    //generate include header file
    lu_header << "#pragma once\n";
    lu_header << declaration::typedef_int->get_code() << "\n";
    if (device_type() == CUDA_GPU || device_type() == ROCM_GPU)
        lu_header << header::cuda->get_code();
    // TODO only include this if half is used
    if (device_type() == CUDA_GPU)
        lu_header << header::cuda_fp16->get_code();

    lu_header << "extern \"C\" int kernel_entry(";
    std::string params = get_kernel_entry_paras(tu);
    lu_header << params;
    lu_header << ");\n";

    lu_header << "extern \"C\" void cuda_init();\n";

    lu_header << "extern \"C\" void cuda_free();\n";

    LanguageUnit_p h =
        std::make_shared<LanguageUnit>("header::nnfusion_rt.h", "#include \"nnfusion_rt.h\"\n");
    projgen->lup_exec->require(h);

    return;
}

void CudaCodegenPass::create_main_file(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_main = std::make_shared<LanguageUnit>("codegen_main");
    projgen->lup_codegen->require(lup_main);
    lup_main->pwd = m_codegen_folder;
    lup_main->write_to = "main_test.cpp";

    auto& lu_main = *lup_main;

    LanguageUnit_p re_main = make_shared<LanguageUnit>("main_include");
    re_main->require(header::stdlib);
    re_main->require(header::stdio);
    re_main->require(header::sstream);
    re_main->require(header::stdexcept);
    re_main->require(header::limits);

    re_main->require(header::cuda_prof_api);
    // re_main->require(header::cuda_fp16);
    re_main->require(macro::CUDA_SAFE_CALL);

    lu_main << "#include \"nnfusion_rt.h\"\n";

    for (auto& it : re_main->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            lu_main << it.second->get_code();

    for (auto& it : re_main->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            lu_main << it.second->get_code() << "\n";

    LanguageUnit h2dcopy("h2dcopy");
    LanguageUnit d2hcopy("d2hcopy");
    LanguageUnit fillval("fillval");

    lu_main << "int main(void)";
    lu_main.block_begin();
    {
        lu_main << "\ncuda_init();\n\n";

        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            //malloc host input arg
            lu_main << "//input argument\n";
            lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                    << "_host, *" << tensor.get_name() << ";\n";

            lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                    << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                    << tensor.get_tensor_layout()->get_size() << "));\n";
            lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ", "
                    << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                    << tensor.get_tensor_layout()->get_size() << "));\n";

            fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size() << "; ++i) "
                    << tensor.get_name() << "_host[i] = 1.0f;\n";

            h2dcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", "
                    << tensor.get_name() << "_host, "
                    << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                    << tensor.get_tensor_layout()->get_size() << ", "
                    << "cudaMemcpyHostToDevice));\n";
        }

        lu_main << "\n//output arguments\n";
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            //malloc host output arg
            lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                    << "_host, *" << tensor.get_name() << ";\n";

            lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                    << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                    << tensor.get_tensor_layout()->get_size() << "));\n";

            if (FLAGS_fextern_result_memory)
            {
                lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                        << " sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << "));\n";
            }
            d2hcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                    << tensor.get_name() << ", "
                    << " sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                    << tensor.get_tensor_layout()->get_size() << ", "
                    << "cudaMemcpyDeviceToHost));\n";
        }

        lu_main << "\n// fill input values\n";
        lu_main << fillval.get_code() << "\n";
        lu_main << h2dcopy.get_code() << "\n";

        vector<string> params;
        for (int i = 0; i < tu->arg.size(); i++)
        {
            auto& tv = tu->arg[i];
            params.push_back(tv->get_name());
        }
        for (int i = 0; i < tu->out.size(); i++)
        {
            auto& tv = tu->out[i];
            if (FLAGS_fextern_result_memory)
                params.push_back(tv->get_name());
            else
                params.push_back("&" + tv->get_name());
        }
        int warm_step = FLAGS_fwarmup_step, test_step = FLAGS_frun_step;
        if (FLAGS_fcodegen_debug)
        {
            warm_step = 0;
            test_step = 1;
        }
        lu_main << "\n//warm up for 5 iters:\n";
        lu_main << "for(int i_=0; i_< " << warm_step << "; i_++)\n";
        lu_main.block_begin();
        lu_main << h2dcopy.get_code();
        lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
        lu_main << d2hcopy.get_code();
        lu_main << "CUDA_SAFE_CALL(cudaDeviceSynchronize()); \n";

        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "printf(\"%s \\n\", \"" << tensor.get_name() << ":\");\n"
                    << "for (int i = 0; i < "
                    << std::min(size_t(10), tensor.get_tensor_layout()->get_size())
                    << "; ++i) printf(\"%e \", (float)" << tensor.get_name() << "_host[i]); "
                    << "\nprintf(\" .. (size = " << tensor.get_tensor_layout()->get_size()
                    << ", ends with %e);\\n\", (float)" << tensor.get_name() << "_host["
                    << tensor.get_tensor_layout()->get_size() - 1 << "]);\n";
        }
        lu_main.block_end();

        lu_main << "\n//GPU time measurement\n";
        lu_main << "float ms_max = std::numeric_limits<float>::min();\n";
        lu_main << "float ms_min = std::numeric_limits<float>::max();\n";
        lu_main << "float ms_total, ms_i;\n";
        lu_main << "cudaEvent_t start, stop, start_i, stop_i;\n";
        lu_main << "cudaEventCreate(&start);\n";
        lu_main << "cudaEventCreate(&stop);\n";
        lu_main << "cudaEventCreate(&start_i);\n";
        lu_main << "cudaEventCreate(&stop_i);\n";

        lu_main << "\n//time measurement\n";
        lu_main << "cudaEventRecord(start);\n\n";
        lu_main << "//kernel call\n";

        lu_main << "int steps = " << test_step << ";\n";
        lu_main << "cudaProfilerStart();\n";
        lu_main << "for (int i_=0; i_<steps; i_++)\n";
        lu_main.block_begin();

        lu_main << "cudaEventRecord(start_i, 0);\n";
        lu_main << h2dcopy.get_code();
        // kernel launch
        lu_main << "kernel_entry(" << join(params, ", ") << ");\n";

        lu_main << "cudaEventRecord(stop_i, 0);\n";
        lu_main << "cudaEventSynchronize(stop_i);\n";
        lu_main << "cudaEventElapsedTime(&ms_i, start_i, stop_i);\n";
        lu_main << "printf(\"Iteration time %f ms\\n\", ms_i);\n";
        lu_main << "if (ms_i > ms_max)  ms_max = ms_i;\n";
        lu_main << "if (ms_i < ms_min) ms_min = ms_i;\n";

        lu_main.block_end();
        lu_main << "cudaProfilerStop();\n";

        lu_main << "//time measurement\n";
        lu_main << "\ncudaEventRecord(stop);\n";
        lu_main << "cudaEventSynchronize(stop);\n";
        lu_main << "cudaEventElapsedTime(&ms_total, start, stop);\n";
        lu_main << "printf(\"Summary: [min, max, mean] = [%f, %f, %f] ms\\n\",  ms_min, ms_max, "
                   "ms_total/steps);\n";
        lu_main << "\n//free context\n";

        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
        }

        if (FLAGS_fextern_result_memory)
        {
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }
        }

        lu_main << "cuda_free();\n\n";

        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }
        //free host output args
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }

        lu_main << "return 0;\n";
    }

    lu_main.block_end();

    return;
}

// todo: add flags for future.
void CudaCodegenPass::create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_cmake = std::make_shared<LanguageUnit>("codegen_cmake");
    projgen->lup_codegen->require(lup_cmake);
    lup_cmake->pwd = m_codegen_folder;
    lup_cmake->write_to = "CMakeLists.txt";

    auto& lu = *lup_cmake;
    lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

SET(SRC "nnfusion_rt.cu" CACHE STRING "codegen source file")
SET(TARGET_NAME "nnfusion_naive_rt" CACHE STRING "codegen target name")
SET(CUDA_ARCH "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" CACHE STRING "target architecture")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")
)";

    if (FLAGS_fkernels_as_files)
    {
        lu << "\nfile(GLOB kernels kernels/*" << m_kernel_suffix << ")\n";

        lu << "list(APPEND SRC ${kernels} shared" << m_kernel_suffix << ")\n";

        lu << "include_directories(${CMAKE_SOURCE_DIR})\n\n";
    }

    lu << "cuda_add_library(${TARGET_NAME} SHARED ${SRC})\n";

    // Prepare submodule
    {
        // add cuda_lib
        lu << nnfusion::codegen::cmake::cuda_lib->get_code();

        if (host_async_manager->num_non_default_stream() > 0)
        {
            // add eigen
            lu << nnfusion::codegen::cmake::eigen->get_code();

            // add threadpool
            lu << nnfusion::codegen::cmake::threadpool->get_code();

            // add threads
            lu << nnfusion::codegen::cmake::threads->get_code();
        }

        if (global_required.count("header::super_scaler") > 0)
        {
            // add super_scaler
            lu << nnfusion::codegen::cmake::super_scaler->get_code();
        }
    }

    lu << R"(
cuda_add_executable(main_test main_test.cpp)   
target_link_libraries(main_test ${TARGET_NAME}) 
)";
    return;
}
