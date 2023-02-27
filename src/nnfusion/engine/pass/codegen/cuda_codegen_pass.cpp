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

#include <regex>

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

DEFINE_bool(fcodegen_debug, false, "Add debug functions in Codegen-ed project.");
DEFINE_bool(fcodegen_debug_half, false, "");
DECLARE_string(fdefault_device);
DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(frt_const_folding);
DECLARE_string(fcuda_init_stream);
DECLARE_bool(fextern_result_memory);
DECLARE_int32(fwarmup_step);
DECLARE_int32(frun_step);
DECLARE_bool(fcustomized_mem_imp);
DECLARE_bool(fhost_entry);
DECLARE_string(fantares_perf_file);
DECLARE_bool(fcodegen_pybind);
DECLARE_bool(ffunction_codegen);
DECLARE_bool(fmulti_shape);

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

    superscaler_enable = global_required.count("header::superscaler") > 0;
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

    if (superscaler_enable)
    {
        std::string superscaler_path = std::string(path) + std::string("/superscaler");
        copy_folder.push_back(superscaler_path);
    }

    if (global_required.count("header::cub") > 0)
    {
        std::string cub_path = std::string(path) + std::string("/cub");
        copy_folder.push_back(cub_path);
    }

    if (global_required.count("declaration::mem_eff_attn") > 0)
    {
        std::string cutlass_path = std::string(path) + std::string("/cutlass");
        copy_folder.push_back(cutlass_path);
    }

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        if (superscaler_enable)
        {
            lu_init_begin << "\nextern \"C\" void cuda_init(const char* resource_dir)\n{\n";
            lu_init_begin << "CUDA_SAFE_CALL(cudaDeviceReset());\n";
            lu_init_begin <<
                R"(int device_id;
int host_id;
sc_init(resource_dir);
sc_get_host_id(&host_id);
sc_get_device_id(&device_id);
printf("[host_id: %d device_id: %d] is running\n", host_id, device_id);
CUDA_SAFE_CALL(cudaSetDevice(device_id));
)";
        }
        else
        {
            if (FLAGS_ffunction_codegen)
                lu_init_begin << "\nextern \"C\" void cuda_init(char* workspace)\n{\n";
            else
                lu_init_begin << "\nextern \"C\" void cuda_init()\n{\n";
            lu_init_begin << "// CUDA_SAFE_CALL(cudaDeviceReset());\n";
        }
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
        lu_exec_init << codegen_global_symbols(tu)->get_code();
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

    if (FLAGS_fcodegen_pybind)
    {
        auto& lu_exec_py_begin = *(projgen->lup_exec_py->begin);
        {
            if (FLAGS_fcodegen_pybind)
            {
                auto params_info = get_kernel_torch_entry_paras(tu);
                lu_exec_py_begin << "\nextern \"C\" void kernel_torch_entry(" << params_info.first
                                 << ")\n{\n";
                lu_exec_py_begin << params_info.second << "\n";
            }
        }

        auto& lu_exec_py_end = *(projgen->lup_exec_py->end);
        {
            lu_exec_py_end << "}\n\n";
        }
    }

    if (FLAGS_fhost_entry)
    {
        fill_exec_host(tu);
    }

    auto& lu_exit_begin = *(projgen->lup_exit->begin);
    {
        lu_exit_begin << "\nextern \"C\" void cuda_free()\n{\n";
        if (superscaler_enable)
        {
            lu_exit_begin <<
                R"(int device_id;
sc_get_device_id(&device_id);
CUDA_SAFE_CALL(cudaSetDevice(device_id));
)";
        }
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        if (superscaler_enable)
            lu_exit_end << "sc_finalize();\n";
        lu_exit_end << "}\n\n";

        if (FLAGS_fcodegen_pybind)
        {
            lu_exit_end << "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n";
            lu_exit_end << "    m.def(\"kernel_entry\", &kernel_torch_entry, \"kernel torch entry "
                           "(CUDA)\");\n";
            lu_exit_end << "    m.def(\"cuda_init\", &cuda_init, \"cuda init (CUDA)\");\n";
            lu_exit_end << "}\n";
        }
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
    if (FLAGS_fcodegen_pybind)
        projgen->lup_codegen->require(header::torch_extension);
    projgen->lup_codegen->require(macro::CUDA_SAFE_CALL);
    projgen->lup_codegen->require(macro::CUDNN_SAFE_CALL);
    projgen->lup_codegen->require(macro::CUBLAS_SAFE_CALL);
    if (!FLAGS_fcodegen_pybind)
        projgen->lup_codegen->require(macro::HALF_MAX);
    projgen->lup_codegen->require(macro::CUDA_HALF_OPERATIONS);
    projgen->lup_codegen->require(macro::TVM_PACK_VALUES);
    projgen->lup_codegen->require(codegen_device_type());
    projgen->lup_codegen->require(codegen_workspace_size(tu));
    return;
}

bool CudaCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

    // collect code
    auto pairs = collect_ins(ctx, tu);
    std::unordered_map<std::string, std::string> replaced_extern_result_memory;
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
            if (gnode && kernel && kernel->is_eliminative())
            {
                (*gnode)["is_eliminative"] = true;
            }
        }

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
                    if (!(*gnode)["is_eliminative"].is_valid_as<bool>())
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

            std::string call_str = fu->get_specialized_function_call(func_name);
            // this hack is to eliminate d2d copy caused by extern result memory
            // we only apply the repalce for non-eliminative ops
            if (FLAGS_fextern_result_memory && gnode &&
                !((*gnode)["is_eliminative"].is_valid_as<bool>()) &&
                !gnode->get_op_ptr()->is_tensor_op())
            {
                auto out_users = gnode->get_output_users(0, false);
                if (gnode->get_output_size() == 1 && out_users.size() == 1 &&
                    !is_ref_tensor(ins, kernel->m_context->outputs[0]))
                {
                    // find the output node along a sequnece of eliminative nodes
                    auto next_node = out_users[0]->get_dst();
                    while (!next_node->get_op_ptr()->is_output() &&
                           (*next_node)["is_eliminative"].is_valid_as<bool>())
                    {
                        out_users = next_node->get_output_users(0, false);
                        if (out_users.size() != 1)
                            break;
                        next_node = out_users[0]->get_dst();
                    }
                    if (next_node->get_op_ptr()->is_output())
                    {
                        std::string in_name = gnode->get_output_tensor(0).get_name();
                        std::string out_name = next_node->get_output_tensor(0).get_name();
                        int pos = call_str.find(", " + in_name);
                        call_str.replace(pos, in_name.size() + 2, ", " + out_name);
                        (*next_node)["is_eliminative"] = true;
                    }
                }
            }

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
            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).first);
            lup_func_calls->unit_vec.push_back(kernel_func_call);
            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).second);
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
                std::string t_threadpool_call =
                    (superscaler_enable && thread_name != "dev0_thread")
                        ? std::string("superscaler_schedule_thread->Schedule(")
                        : std::string("schedule_thread_pool->Schedule(");
                t_threadpool_call += (std_thread_func_name + std::string(");\n"));
                lu_new_caller << t_threadpool_call;
            }

            LanguageUnit_p begin =
                std::make_shared<LanguageUnit>(lup_func_calls->symbol + "_new_caller_block_begin");
            auto& lu_begin = *begin;
            {
                lu_begin << "extern \"C\" void " << thread_name << "(";
                lu_begin << thread_call_paras << ")\n{\n";
                if (superscaler_enable)
                {
                    lu_begin << R"(int device_id;
sc_get_device_id(&device_id);
CUDA_SAFE_CALL(cudaSetDevice(device_id));
)";
                }
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

            // this tensor will be shared buffer with other one, skip init here.
            if (gnode && gnode->is_constant() && (*gnode)["shared_tensor"].is_valid_as<bool>())
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
                 else
                 {
                     return a.first > b.first;
                 }
             });
    }

    return pairs;
}

std::string CudaCodegenPass::get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu,
                                                    bool is_host)
{
    unordered_set<string> allocated;
    vector<string> params;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = element::get_backend_cstring(tv->get_element_type());
        stringstream ss;
        ss << type << "* " << tv->get_name();
        if (is_host)
        {
            ss << "_host";
        }
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }

    for (int i = 0; i < tu->out.size(); i++)
    {
        auto tv = tu->out[i];
        string type = element::get_backend_cstring(tv->get_element_type());
        stringstream ss;
        if (FLAGS_fextern_result_memory || FLAGS_fhost_entry)
            ss << type << "* " << tv->get_name();
        else
            ss << type << "** " << tv->get_name();
        if (is_host)
        {
            ss << "_host";
        }
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }
    return join(params, ", ");
}

std::pair<std::string, std::string>
    CudaCodegenPass::get_kernel_torch_entry_paras(std::shared_ptr<TranslationUnit> tu)
{
    std::string paras, refs;
    unordered_set<string> allocated;
    vector<string> params, references;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = element::get_backend_cstring(tv->get_element_type());
        stringstream ss1, ss2;
        ss1 << "torch::Tensor " << tv->get_name() << "_ts";
        ss2 << type << "* " << tv->get_name() << " = " << tv->get_name() << "_ts.data_ptr<" << type
            << ">();";
        allocated.insert(tv->get_name());
        params.push_back(ss1.str());
        references.push_back(ss2.str());
    }

    for (int i = 0; i < tu->out.size(); i++)
    {
        auto tv = tu->out[i];
        string type = element::get_backend_cstring(tv->get_element_type());
        stringstream ss1, ss2;
        ss1 << "torch::Tensor " << tv->get_name() << "_ts";
        if (FLAGS_fextern_result_memory || FLAGS_fhost_entry)
            ss2 << type << "* " << tv->get_name() << " = " << tv->get_name() << "_ts.data_ptr<"
                << type << ">();";
        else
            ss2 << type << "** " << tv->get_name() << " = " << tv->get_name() << "_ts.data_ptr<"
                << type << ">();";
        allocated.insert(tv->get_name());
        params.push_back(ss1.str());
        references.push_back(ss2.str());
    }
    paras = join(params, ", ");
    refs = join(references, "\n");
    return std::make_pair(paras, refs);
}

std::string CudaCodegenPass::get_kernel_entry_args(std::shared_ptr<TranslationUnit> tu,
                                                   bool is_host)
{
    vector<string> args;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto& tv = tu->arg[i];
        auto name = tv->get_name();
        if (is_host)
        {
            name = name + "_host";
        }
        args.push_back(name);
    }
    for (int i = 0; i < tu->out.size(); i++)
    {
        auto& tv = tu->out[i];
        auto name = tv->get_name();
        if (is_host)
        {
            name = name + "_host";
        }
        if (FLAGS_fextern_result_memory || FLAGS_fhost_entry)
            args.push_back(name);
        else
            args.push_back("&" + name);
    }
    return join(args, ", ");
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
                    string type = element::get_backend_cstring(input->get_element_type());
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
                        string type = element::get_backend_cstring(output->get_element_type());
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

    auto mem_ref = codegen_mem_ref(ins);
    if (mem_ref != nullptr)
        lu << codegen_mem_ref(ins)->get_code();

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
        if ((*(ins->getGNode()))["is_eliminative"].is_valid_as<bool>())
        {
            lu << "// eliminated: " << func_call;
        }
        // // todo: this hack is to eliminate d2d copy caused by extern result memory
        // else if (FLAGS_fextern_result_memory && gnode && gnode->get_op_ptr()->is_output())
        // {
        //     lu << "// eliminated: " << func_call;
        // }

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

    std::string member_name = gnode ? "(" + gnode->get_member_name() + ")" : "";
    if (FLAGS_fcodegen_debug && gnode && kernel && !gnode->get_op_ptr()->is_output())
    {
        for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
        {
            if (element::get_backend_cstring(kernel->m_context->outputs[i]->get_element_type()) !=
                    "float" &&
                element::get_backend_cstring(kernel->m_context->outputs[i]->get_element_type()) !=
                    "half")
                continue;
            auto out_name = kernel->m_context->output_names[i];

            lu << "Debug(\"" << node_name << ", " << out_name << member_name << "\", " << out_name
               << ", \"" << join(kernel->m_context->input_names) << "\", "
               << kernel->m_context->outputs[i]->size(false) << ");\n";
        }
        lu.require(codegen::helper::debug);
    }

    if (FLAGS_fcodegen_debug_half)
    {
        for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
        {
            auto outshape = kernel->m_context->outputs[i]->get_shape();
            auto out_name = kernel->m_context->output_names[i];
            if (element::get_backend_cstring(kernel->m_context->outputs[i]->get_element_type()) ==
                "half")
            {
                int grids, blocks, bound;
                CudaCodegenPass::compute_best_config(outshape, grids, blocks, bound);
                if (grids == 1)
                {
                    lu << "Convert_half_float_Call0(dim3(" << grids << ", 1, 1), dim3(" << blocks
                       << ", 1, 1), 0, 0, " << out_name << ", fp32tensors, " << bound << ");\n";
                }
                else
                {
                    lu << "Convert_half_float_Call1(dim3(" << grids << ", 1, 1), dim3(" << blocks
                       << ", 1, 1), 0, 0, " << out_name << ", fp32tensors, " << blocks << ", "
                       << bound << ");\n";
                }

                lu << "Debug(\"" << node_name << ", " << out_name << member_name << "_f32\", "
                   << "fp32tensors, \"" << join(kernel->m_context->input_names) << "\", "
                   << kernel->m_context->outputs[i]->size(false) << ");\n";
                lu << "CUDA_SAFE_CALL(cudaMemset((void*)fp32tensors, 0, "
                   << max_tensor_size <<"));\n";
            }
            else if (element::get_backend_cstring(
                         kernel->m_context->outputs[i]->get_element_type()) == "float")
            {
                lu << "Debug(\"" << node_name << ", " << out_name << member_name << "\", "
                   << out_name << ", \"" << join(kernel->m_context->input_names) << "\", "
                   << kernel->m_context->outputs[i]->size(false) << ");\n";
            }
        }

        lu.require(codegen::helper::debug);
        lu.require(codegen::helper::cuda_half_debug);
    }

    return _lu;
}

bool CudaCodegenPass::collect_stream(std::shared_ptr<InterpreterContext> ctx,
                                     std::shared_ptr<TranslationUnit> tu)
{
    std::regex r(R"(CUDA_SAFE_CALL\(cudaSetDevice\(\d)");

    //stream
    NNFUSION_CHECK_NOT_NULLPTR(device_async_manager);
    if (device_async_manager && device_async_manager->num_stream() > 0)
    {
        auto stream_decl = device_async_manager->emit_stream_decl();
        auto stream_init = device_async_manager->emit_stream_init();
        auto stream_destroy = device_async_manager->emit_stream_destroy();

        string stream_init_code_old = stream_init->get_code();
        string stream_destroy_code_old = stream_destroy->get_code();
        string stream_init_code =
            (superscaler_enable ? std::regex_replace(stream_init_code_old, r, "// $0")
                                : stream_init_code_old);
        string stream_destroy_code =
            (superscaler_enable ? std::regex_replace(stream_destroy_code_old, r, "// $0")
                                : stream_destroy_code_old);
        LanguageUnit_p stream_init_lu(
            new LanguageUnit(stream_init->get_symbol(), stream_init_code));
        LanguageUnit_p stream_destroy_lu(
            new LanguageUnit(stream_destroy->get_symbol(), stream_destroy_code));

        stream_init_lu->require(stream_decl);
        add_init_and_exit_pair(stream_init_lu, stream_destroy_lu);
    }

    //event
    if (device_async_manager && device_async_manager->num_event() > 0)
    {
        auto event_decl = device_async_manager->emit_event_decl();
        auto event_init = device_async_manager->emit_event_init();
        auto event_destroy = device_async_manager->emit_event_destroy();

        string event_init_code_old = event_init->get_code();
        string event_destroy_code_old = event_destroy->get_code();
        string event_init_code =
            (superscaler_enable ? std::regex_replace(event_init_code_old, r, "// $0")
                                : event_init_code_old);
        string event_destroy_code =
            (superscaler_enable ? std::regex_replace(event_destroy_code_old, r, "// $0")
                                : event_destroy_code_old);

        LanguageUnit_p event_init_lu(new LanguageUnit(event_init->get_symbol(), event_init_code));
        LanguageUnit_p event_destroy_lu(
            new LanguageUnit(event_destroy->get_symbol(), event_destroy_code));

        event_init_lu->require(event_decl);
        add_init_and_exit_pair(event_init_lu, event_destroy_lu);
    }

    return true;
}

bool CudaCodegenPass::collect_mem(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;
    auto mem_pair = create_init_and_exit_pair<LanguageUnitwithVec, LanguageUnitwithVec>("MEM_ALLOC",
                                                                                        "MEM_FREE");
    auto lup_mem_alloc = mem_pair.first;
    auto lup_mem_free = mem_pair.second;
    auto& allocator_list = tu->memory_allocator_factory->get_allocator_list();

    size_t total_alloc = 0;
    max_tensor_size = 0;
    for (const auto& allocator : allocator_list)
    {
        total_alloc += allocator.second->max_allocated();
        if (allocator.second->max_alloc_unit() > max_tensor_size)
            max_tensor_size = allocator.second->max_allocated();
    }

    max_tensor_size *= 2;

    if (FLAGS_fcodegen_debug_half)
    {
        LanguageUnit_p fp32tensors =
            std::make_shared<LanguageUnit>("fp32tensors",
                                           "CUDA_SAFE_CALL(cudaMalloc((void**)&fp32tensors," +
                                               std::to_string(max_tensor_size) + "));\n");
        LanguageUnit_p fp32tensors_decl =
            std::make_shared<LanguageUnit>("fp32tensors_decl", "float* fp32tensors;\n");
        lup_mem_alloc->unit_vec.push_back(fp32tensors);
        lup_mem_alloc->require(fp32tensors_decl);
    }

    LanguageUnit_p total = std::make_shared<LanguageUnit>(
        "total_memory", "// total memory:" + to_string(total_alloc) + "\n");
    lup_mem_alloc->unit_vec.push_back(total);

    std::regex r(R"(CUDA_SAFE_CALL\(cudaSetDevice\(\d)");

    size_t offset = 0;
    for (const auto& allocator : allocator_list)
    {
        auto init = allocator.second->emit_memory_init();
        auto alloc = allocator.second->emit_memory_alloc();
        auto free = allocator.second->emit_memory_free();

        string alloc_code_old = alloc->get_code();
        string free_code_old = free->get_code();

        string alloc_code =
            (superscaler_enable ? std::regex_replace(alloc_code_old, r, "// $0") : alloc_code_old);
        string free_code =
            (superscaler_enable ? std::regex_replace(free_code_old, r, "// $0") : free_code_old);

        LanguageUnit_p alloc_lu(new LanguageUnit(alloc->get_symbol(), alloc_code));
        LanguageUnit_p free_lu(new LanguageUnit(free->get_symbol(), free_code));

        if (FLAGS_ffunction_codegen)
        {
            auto mempool_offset = allocator.second->emit_memory_pool_offset(offset);
            offset += allocator.second->max_allocated();
            lup_mem_alloc->unit_vec.push_back(mempool_offset);
        }
        lup_mem_alloc->unit_vec.push_back(alloc_lu);
        lup_mem_alloc->require(init);
        lup_mem_free->unit_vec.push_back(free_lu);
        lup_mem_free->require(init);
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
        if (superscaler_enable)
            projgen->lup_codegen->require(declaration::superscaler_schedule_thread);
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

        if (superscaler_enable)
        {
            auto superscaler_schedule_thread_pair =
                create_init_and_exit_pair<LanguageUnit, LanguageUnit>(
                    "init_superscaler_schedule_thread", "del_superscaler_schedule_thread");
            auto lup_superscaler_schedule_thread_pair_init = superscaler_schedule_thread_pair.first;
            auto lup_superscaler_schedule_thread_pair_del = superscaler_schedule_thread_pair.second;

            auto& lu_superscaler_schedule_thread_init = *lup_superscaler_schedule_thread_pair_init;
            {
                lu_superscaler_schedule_thread_init
                    << "superscaler_schedule_thread = new concurrency::NumaAwareThreadPool(1,1);\n";
            }
            auto& lu_superscaler_schedule_thread_del = *lup_superscaler_schedule_thread_pair_del;
            {
                lu_superscaler_schedule_thread_del << "delete superscaler_schedule_thread;\n";
            }
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
        LanguageUnit_p device_sync = get_sync();
        projgen->lup_init->unit_vec.push_back(device_sync);
    }

    if (FLAGS_fcodegen_pybind)
    {
        for (auto item : projgen->lup_exec->unit_vec)
        {
            nnfusion::LanguageUnit_p py_item =
                std::make_shared<LanguageUnit>(item->symbol, item->get_code());
            projgen->lup_exec_py->unit_vec.push_back(py_item);
        }
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
                        << element::get_backend_cstring(tu->arg[i]->get_element_type()) << "\n";
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
                        << element::get_backend_cstring(tu->out[i]->get_element_type()) << "\n";
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
    {
        lu_header << header::cuda_fp16->get_code();
        lu_header << header::cuda_mma->get_code();
    }
    lu_header << "extern \"C\" int get_device_type();\n";
    lu_header << "extern \"C\" size_t get_workspace_size();\n";
    lu_header << "extern \"C\" int kernel_entry";
    if (FLAGS_fhost_entry)
        lu_header << "_host";
    std::string params = get_kernel_entry_paras(tu, FLAGS_fhost_entry);
    lu_header << "(" << params << ");\n";

    if (superscaler_enable)
        lu_header << "extern \"C\" void cuda_init(const char*);\n";
    else if (FLAGS_ffunction_codegen)
        lu_header << "extern \"C\" void cuda_init(char* workspace_size);\n";
    else
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

    LanguageUnit fillval("fillval");

    lu_main << "int main(int argc, char *argv[])";
    lu_main.block_begin();
    {
        if (superscaler_enable)
        {
            lu_main << "\nif(!argv[1]) {throw std::runtime_error(\"superscaler resource dir is not "
                       "given!\"); }\n\n";
            lu_main << "\ncuda_init(argv[1]);\n\n";
        }
        else if (FLAGS_ffunction_codegen)
        {
            lu_main << "\nchar* workspace;\n";
            lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&workspace, get_workspace_size()));\n";
            lu_main << "cuda_init(workspace);\n\n";
        }
        else
            lu_main << "\ncuda_init();\n\n";

        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            //malloc host input arg
            lu_main << "//input argument\n";
            lu_main << element::get_backend_cstring(tensor.get_element_type()) << "* "
                    << tensor.get_name() << "_host, *" << tensor.get_name() << ";\n";

            lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                    << "_host, sizeof(" << element::get_backend_cstring(tensor.get_element_type())
                    << ")* " << tensor.get_tensor_layout()->get_size() << "));\n";
            if (!FLAGS_fhost_entry)
            {
                lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ", "
                        << "sizeof(" << element::get_backend_cstring(tensor.get_element_type())
                        << ") * " << tensor.get_tensor_layout()->get_size() << "));\n";
            }
            fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size() << "; ++i) "
                    << tensor.get_name() << "_host[i] = 1.0f;\n";
        }

        lu_main << "\n//output arguments\n";
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            //malloc host output arg
            lu_main << element::get_backend_cstring(tensor.get_element_type()) << "* "
                    << tensor.get_name() << "_host, *" << tensor.get_name() << ";\n";

            lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                    << "_host, sizeof(" << element::get_backend_cstring(tensor.get_element_type())
                    << ") * " << tensor.get_tensor_layout()->get_size() << "));\n";

            if (FLAGS_fextern_result_memory && !FLAGS_fhost_entry)
            {
                lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                        << " sizeof(" << element::get_backend_cstring(tensor.get_element_type())
                        << ") * " << tensor.get_tensor_layout()->get_size() << "));\n";
            }
        }

        lu_main << "\n// fill input values\n";
        lu_main << fillval.get_code() << "\n";

        int warm_step = FLAGS_fwarmup_step, test_step = FLAGS_frun_step;
        if (FLAGS_fcodegen_debug || FLAGS_fcodegen_debug_half)
        {
            warm_step = 0;
            test_step = 1;
        }
        lu_main << "\n//warm up for 5 iters:\n";
        lu_main << "for(int i_=0; i_< " << warm_step << "; i_++)\n";
        lu_main.block_begin();
        std::string args;
        if (FLAGS_fhost_entry)
        {
            args = get_kernel_entry_args(tu, true);
            lu_main << "kernel_entry_host(" << args << ");\n";
        }
        else
        {
            args = get_kernel_entry_args(tu, false);
            lu_main << get_h2dcopy(tu)->get_code();
            lu_main << "kernel_entry(" << args << ");\n";
            lu_main << get_d2hcopy(tu)->get_code();
            lu_main << get_sync()->get_code();
        }
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
        lu_main << "cudaEvent_t start_i, stop_i;\n";
        lu_main << "cudaEventCreate(&start_i);\n";
        lu_main << "cudaEventCreate(&stop_i);\n";

        lu_main << "\n//time measurement\n";
        lu_main << "ms_total = 0;\n\n";
        lu_main << "//kernel call\n";

        lu_main << "int steps = " << test_step << ";\n";
        // lu_main << get_h2dcopy(tu)->get_code();
        lu_main << "cudaProfilerStart();\n";
        lu_main << "for (int i_=0; i_<steps; i_++)\n";
        lu_main.block_begin();

        lu_main << "cudaEventRecord(start_i, 0);\n";
        if (FLAGS_fhost_entry)
        {
            lu_main << "kernel_entry_host(" << args << ");\n";
        }
        else
        {
            lu_main << get_h2dcopy(tu)->get_code();
            lu_main << "kernel_entry(" << args << ");\n";
            // lu_main << get_d2hcopy(tu)->get_code();
            // lu_main << get_sync()->get_code();
        }

        lu_main << "cudaEventRecord(stop_i, 0);\n";
        lu_main << "cudaEventSynchronize(stop_i);\n";
        lu_main << "cudaEventElapsedTime(&ms_i, start_i, stop_i);\n";
        lu_main << "printf(\"Iteration time %f ms\\n\", ms_i);\n";
        lu_main << "ms_total += ms_i;\n";
        lu_main << "if (ms_i > ms_max)  ms_max = ms_i;\n";
        lu_main << "if (ms_i < ms_min) ms_min = ms_i;\n";

        lu_main.block_end();
        lu_main << "cudaProfilerStop();\n";

        lu_main << "\n//time measurement\n";
        lu_main << "printf(\"Summary: [min, max, mean] = [%f, %f, %f] ms\\n\",  ms_min, ms_max, "
                   "ms_total / steps);\n";
        lu_main << "\n//free context\n";

        if (!FLAGS_fhost_entry)
        {
            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }
        }
        if (FLAGS_fextern_result_memory && !FLAGS_fhost_entry)
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
SET(CUDA_ARCH "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86" CACHE STRING "target architecture")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --prec-sqrt=false --ftz=true --prec-div=false -fmad=true")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
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

        if (superscaler_enable)
        {
            // add superscaler
            lu << nnfusion::codegen::cmake::superscaler_cuda->get_code();
        }

        if (global_required.count("header::cub") > 0)
        {
            lu << nnfusion::codegen::cmake::cub->get_code();
        }

        if (global_required.count("declaration::mem_eff_attn") > 0)
        {
            lu << nnfusion::codegen::cmake::cutlass->get_code();
        }
    }

    lu << R"(
cuda_add_executable(main_test main_test.cpp)
target_link_libraries(main_test ${TARGET_NAME})
)";
    return;
}

LanguageUnit_p CudaCodegenPass::get_d2hcopy(std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p d2hcopy = std::make_shared<LanguageUnit>("d2hcopy");
    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        *d2hcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                 << tensor.get_name() << ", "
                 << " sizeof(" << element::get_backend_cstring(tensor.get_element_type()) << ") * "
                 << tensor.get_tensor_layout()->get_size() << ", "
                 << "cudaMemcpyDeviceToHost));\n";
    }
    return d2hcopy;
}

LanguageUnit_p CudaCodegenPass::get_h2dcopy(std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p h2dcopy = std::make_shared<LanguageUnit>("h2dcopy");

    for (size_t i = 0; i < tu->arg.size(); i++)
    {
        auto& tensor = *tu->arg[i];
        *h2dcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name()
                 << "_host, "
                 << "sizeof(" << element::get_backend_cstring(tensor.get_element_type()) << ") * "
                 << tensor.get_tensor_layout()->get_size() << ", "
                 << "cudaMemcpyHostToDevice));\n";
    }
    return h2dcopy;
}

LanguageUnit_p CudaCodegenPass::get_sync()
{
    return std::make_shared<LanguageUnit>("device_sync",
                                          "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n");
}
void CudaCodegenPass::fill_exec_host(std::shared_ptr<TranslationUnit> tu)
{
    auto lup_exec_host = std::make_shared<CodegenMainBlockUnit>("codegen_exec_host");
    projgen->lup_codegen->require(lup_exec_host);
    projgen->lup_exit->require(lup_exec_host);
    lup_exec_host->require(projgen->lup_exec);

    auto& lu_exec_host_begin = *(lup_exec_host->begin);
    {
        std::string params = get_kernel_entry_paras(tu, true);
        lu_exec_host_begin << "\nint kernel_entry_host(" << params << ")\n{\n";
    }

    auto& lu_exec_host_end = *(lup_exec_host->end);
    {
        lu_exec_host_end << "return 0;\n";
        lu_exec_host_end << "}\n\n";
    }

    auto& lu_exec_host_vec = lup_exec_host->unit_vec;
    lu_exec_host_vec.push_back(get_h2dcopy(tu));
    // lu_exec_host_vec.push_back(get_sync());
    LanguageUnit_p kernel_entry_call = std::make_shared<LanguageUnit>("kernel_entry_call");
    *kernel_entry_call << "kernel_entry(" << get_kernel_entry_args(tu) << ");\n";
    lu_exec_host_vec.push_back(kernel_entry_call);
    // lu_exec_host_vec.push_back(get_sync());
    lu_exec_host_vec.push_back(get_d2hcopy(tu));
    lu_exec_host_vec.push_back(get_sync());
}

void CudaCodegenPass::compute_best_config(nnfusion::Shape outshape,
                                          int& grids,
                                          int& blocks,
                                          int& bound)
{
    uint32_t num_ele = static_cast<uint32_t>(nnfusion::shape_size(outshape));
    for (int i = 512; i >= 64; i >>= 1)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    for (int i = 512; i >= 32; i--)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    if (num_ele < 32)
        grids = 1, blocks = num_ele, bound = 0;
    else
        grids = (num_ele + 255) / 256, blocks = 256, bound = 1;
}

bool CudaMultiCodegenPassPre::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    initialize(ctx, tu);
    NNFUSION_CHECK(collect_mem(ctx, tu));
    NNFUSION_CHECK(collect_stream(ctx, tu));
    NNFUSION_CHECK(collect_funcs(ctx, tu));
    NNFUSION_CHECK(modify_codegen());
    return true;
}
