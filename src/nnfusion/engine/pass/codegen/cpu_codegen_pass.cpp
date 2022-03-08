// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "codegen_langunit.hpp"
#include "codegenerator_helper.hpp"
#include "cpu_codegen_pass.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cpu/reference/reference_common.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

DEFINE_int32(fnuma_node_num, 1, "");
DEFINE_int32(fthread_num_per_node, 0, "");
DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(frt_const_folding);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fcustomized_mem_imp);
DECLARE_bool(ffunction_codegen);

void CpuCodegenPass::set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
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
            need_intra_node_threadpool |= kernel->is_parallelism();
            for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
            {
                global_required.insert(it.second->symbol);
            }
        }
    }
    numa_node_num = FLAGS_fnuma_node_num;
    if (!host_async_manager || host_async_manager->num_non_default_stream() == 0)
    {
        numa_node_num = 1;
    }
    return;
}

void CpuCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                std::shared_ptr<TranslationUnit> tu)
{
    set_global_member(ctx, tu);

    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.cpp";
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

    if (need_intra_node_threadpool ||
        (host_async_manager && host_async_manager->num_non_default_stream() > 0))
    {
        std::string eigen_path = std::string(path) + std::string("/eigen");
        copy_folder.push_back(eigen_path);
        std::string threadpool_path = std::string(path) + std::string("/threadpool");
        copy_folder.push_back(threadpool_path);
    }

    if (global_required.count("header::mlas") > 0)
    {
        std::string mlas_path = std::string(path) + std::string("/mlas");
        copy_folder.push_back(mlas_path);
    }

    if (global_required.count("header::cblas") > 0)
    {
        std::string mkl_path = std::string(path) + std::string("/mkl");
        copy_folder.push_back(mkl_path);
    }

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        if (FLAGS_ffunction_codegen)
            lu_init_begin << "\nextern \"C\" void cpu_init(char* workspace)\n{\n";
        else
            lu_init_begin << "\nextern \"C\" void cpu_init()\n{\n";
    }

    auto& lu_init_end = *(projgen->lup_init->end);
    {
        lu_init_end << "}\n";
    }

    auto& lu_exec_begin = *(projgen->lup_exec->begin);
    {
        std::string params = get_kernel_entry_paras(tu);
        lu_exec_begin << "extern \"C\" int kernel_entry(" << params << ")\n{\n";
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
        lu_exec_end << "}\n";
    }

    auto& lu_exit_begin = *(projgen->lup_exit->begin);
    {
        lu_exit_begin << "extern \"C\" void cpu_free()\n{\n";
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        lu_exit_end << "}\n";
    }
    //add requirements
    projgen->lup_codegen->require(codegen_device_type());
    projgen->lup_codegen->require(codegen_workspace_size(tu));
    // add component
    // create_graph_config(ctx, tu);
    create_header_file(ctx, tu);
    create_main_file(ctx, tu);
    create_cmake_file(ctx, tu);

    return;
}

// todo: add flags for future.
void CpuCodegenPass::create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_cmake = std::make_shared<LanguageUnit>("codegen_cmake");
    projgen->lup_codegen->require(lup_cmake);
    lup_cmake->pwd = m_codegen_folder;
    lup_cmake->write_to = "CMakeLists.txt";

    auto& lu = *lup_cmake;
    lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

SET(SRC "nnfusion_rt.cpp" CACHE STRING "codegen source file")
SET(TARGET_NAME "nnfusion_cpu_rt" CACHE STRING "codegen target name")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++11 -O3 -march=native -pthread")
)";

    if (FLAGS_fkernels_as_files)
    {
        lu << "\nfile(GLOB kernels kernels/*" << m_kernel_suffix << ")\n";
        lu << "list(APPEND SRC ${kernels} shared" << m_kernel_suffix << ")\n";
        lu << "include_directories(${CMAKE_SOURCE_DIR})\n\n";
    }

    lu << "add_library(${TARGET_NAME} ${SRC})\n";

    // Prepare submodule
    {
        if (global_required.count("header::cblas") > 0)
        {
            // add cblas
            lu << nnfusion::codegen::cmake::cblas->get_code();
        }

        if (need_intra_node_threadpool || host_async_manager->num_non_default_stream() > 0)
        {
            // add eigen
            lu << nnfusion::codegen::cmake::eigen->get_code();

            // add threadpool
            lu << nnfusion::codegen::cmake::threadpool->get_code();
        }

        if (global_required.count("header::mlas") > 0)
        {
            // add mlas
            lu << nnfusion::codegen::cmake::mlas->get_code();
        }

        // add threads
        lu << nnfusion::codegen::cmake::threads->get_code();
    }

    lu << R"(
add_executable(main_test main_test.cpp)   
target_link_libraries(main_test ${TARGET_NAME}) 

)";
    return;
}

void CpuCodegenPass::create_main_file(std::shared_ptr<InterpreterContext> ctx,
                                      std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_main = std::make_shared<LanguageUnit>("codegen_main");
    projgen->lup_codegen->require(lup_main);
    lup_main->pwd = m_codegen_folder;
    lup_main->write_to = "main_test.cpp";

    auto& lu_main = *lup_main;

    LanguageUnit_p re_main = make_shared<LanguageUnit>("main_include");
    re_main->require(header::stdio);
    re_main->require(header::stdlib);
    re_main->require(header::sstream);
    re_main->require(header::stdexcept);

    re_main->require(header::chrono);

    lu_main << "#include \"nnfusion_rt.h\"\n";

    for (auto& it : re_main->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            lu_main << it.second->get_code();

    lu_main << "using Clock = std::chrono::high_resolution_clock;\n\n";

    LanguageUnit fillval("fillval");

    lu_main << "int main(void)";
    lu_main.block_begin();
    {
        lu_main << "\ncpu_init();\n\n";

        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            //malloc host input arg
            lu_main << "//input argument\n";
            lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                    << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                    << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                    << tensor.get_tensor_layout()->get_size() << ");\n";

            fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size() << "; ++i) "
                    << tensor.get_name() << "_host[i] = 1.0f;\n";
        }

        lu_main << "\n//output arguments\n";
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            //malloc host output arg
            if (FLAGS_fextern_result_memory)
            {
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << ");\n ";
            }
            else
            {
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host;\n";
            }
        }
        lu_main << "\n//fill input values\n";
        lu_main << fillval.get_code();
        std::string args = get_kernel_entry_args(tu, true);

        lu_main << "\n//warm up\n";
        lu_main << "int warm_steps = 5;\n";
        lu_main << "for(int i_=0; i_<warm_steps; i_++)\n";
        lu_main.block_begin();
        // kernel launch
        lu_main << "kernel_entry(" << args << ");\n";
        lu_main.block_end();

        lu_main << "\n//time measurement\n";
        lu_main << "auto t_start = Clock::now();\n\n";
        lu_main << "//kernel call\n";
        lu_main << "int test_steps = 100;\n";
        lu_main << "for(int i_=0; i_<test_steps; i_++)\n";
        lu_main.block_begin();
        // kernel launch
        lu_main << "kernel_entry(" << args << ");\n";
        lu_main.block_end();

        lu_main << "\n//time measurement\n";
        lu_main << "auto t_end = Clock::now();\n";
        lu_main << "std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;\n\n";

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

        lu_main << "\nprintf(\"function execution time: %f ms\\n\", fp_ms.count()/test_steps);\n";
        lu_main << "\n//free context\n";
        lu_main << "cpu_free();\n";

        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "free(" << tensor.get_name() << "_host);\n";
        }
        // free host output args
        if (FLAGS_fextern_result_memory)
        {
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "free(" << tensor.get_name() << "_host);\n";
            }
        }
        lu_main << "\nreturn 0;\n";
    }
    lu_main.block_end();

    return;
}

void CpuCodegenPass::create_header_file(std::shared_ptr<InterpreterContext> ctx,
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
    // if (device_type() == CUDA_GPU || device_type() == ROCM_GPU)
    //     lu_header << header::cuda->get_code();
    lu_header << "extern \"C\" int get_device_type();\n";
    lu_header << "extern \"C\" int get_workspace_size();\n";
    lu_header << "extern \"C\" int kernel_entry(";
    std::string params = get_kernel_entry_paras(tu);
    lu_header << params;
    lu_header << ");\n";
    if (FLAGS_ffunction_codegen)
        lu_header << "extern \"C\" void cpu_init(char* workspace);\n";
    else
        lu_header << "extern \"C\" void cpu_init();\n";
    lu_header << "extern \"C\" void cpu_free();\n";

    LanguageUnit_p h =
        std::make_shared<LanguageUnit>("header::nnfusion_rt.h", "#include \"nnfusion_rt.h\"\n");
    projgen->lup_exec->require(h);

    return;
}

bool CpuCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

    // collect code
    auto pairs = collect_ins(ctx, tu);
    for (size_t i = 0; i < pairs.size(); i++)
    {
        int numa_node = i % numa_node_num;
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

        size_t cpu_func_count = 0;
        for (auto ins : it.second)
        {
            auto kernel = ins->getKernel();
            auto gnode = ins->getGNode();
            auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string body_str = fu->body_unit->get_code();
            string func_name = fu->name_unit->get_code();
            body_str = fu->signature_unit->get_code() + body_str;
            if (!body_str.empty())
            {
                if (kernel->is_static_function() ||
                    kernel_func_defs.find(body_str) == kernel_func_defs.end())
                {
                    if (!kernel->is_eliminative())
                    {
                        auto kernel_func_def = codegenerator::CPUFunctionFile::convert_from(kernel);

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

            std::string function_call;
            if (thread_name == "default_thread")
            {
                function_call = func_name;
                string call_str = fu->call_unit->get_code();
                if (kernel->is_parallelism())
                {
                    std::string threadpool_param = "worker_thread_pool->GetRawThreadPool(), ";
                    call_str.insert(1, threadpool_param);
                }
                function_call += call_str;
            }
            else
            {
                string call_str = fu->call_unit->get_code();
                if (kernel->is_parallelism())
                {
                    std::string threadpool_param = "worker_thread_pool->GetRawThreadPool(";
                    threadpool_param += (std::to_string(numa_node) + std::string("), "));
                    call_str.insert(1, threadpool_param);
                    function_call = func_name;
                    function_call += call_str;
                }
                else
                {
                    call_str.insert(1, func_name + std::string(", "));
                    std::string std_func_name =
                        std::string("func") + std::to_string(cpu_func_count);
                    std::string std_func_call = std::string("auto ") + std_func_name +
                                                std::string(" = std::bind") + call_str;
                    function_call = std_func_call;
                    std::string threadpool_call = std::string("worker_thread_pool->ScheduleSync(");
                    threadpool_call +=
                        (std_func_name + std::string(", ") + std::to_string(numa_node) + ");\n");
                    function_call += threadpool_call;
                    ++cpu_func_count;
                }
            }

            LanguageUnit_p kernel_func_call = func_call_codegen(ins, func_call_only, function_call);
            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).first);
            lup_func_calls->unit_vec.push_back(kernel_func_call);
            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).second);
            ++cpu_func_count;
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

bool CpuCodegenPass::modify_codegen()
{
    if (global_required.count("header::eigen_spatial_convolution") > 0)
    {
        projgen->lup_init->unit_vec.push_back(std::make_shared<LanguageUnit>(
            "OMP_NUM_THREADS", "setenv(\"OMP_NUM_THREADS\", \"1\", true);\n"));
    }

    // multi-thread
    if (need_intra_node_threadpool ||
        (host_async_manager && host_async_manager->num_non_default_stream() > 0))
    {
        projgen->lup_codegen->require(header::threadpool);
        projgen->lup_codegen->require(declaration::worker_thread_pool);
        // worker thread pool
        auto worker_thread_pool_pair = create_init_and_exit_pair<LanguageUnit, LanguageUnit>(
            "init_worker_thread_pool", "del_worker_thread_pool");
        auto lup_worker_thread_pool_init = worker_thread_pool_pair.first;
        auto lup_worker_thread_pool_del = worker_thread_pool_pair.second;

        auto& lu_worker_thread_pool_init = *lup_worker_thread_pool_init;
        {
            lu_worker_thread_pool_init
                << "worker_thread_pool = new concurrency::NumaAwareThreadPool(" << numa_node_num
                << ", " << FLAGS_fthread_num_per_node << ");\n";
        }
        auto& lu_worker_thread_pool_del = *lup_worker_thread_pool_del;
        {
            lu_worker_thread_pool_del << "delete worker_thread_pool;\n";
        }
    }

    if (host_async_manager && host_async_manager->num_non_default_stream() > 0)
    {
        projgen->lup_codegen->require(declaration::schedule_thread_pool);
    }

    if (host_async_manager &&
        (host_async_manager->num_non_default_stream() > 0 || host_async_manager->num_event() > 0))
    {
        projgen->lup_codegen->require(header::barrier);
    }
    if (host_async_manager && host_async_manager->num_non_default_stream() > 0)
    {
        // default barrier
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

        // init barrier
        LanguageUnit_p init_barrier_decl =
            std::make_shared<LanguageUnit>("declaration::init_barrier_decl");
        projgen->lup_codegen->require(init_barrier_decl);
        auto& lu_init_barrier_decl = *init_barrier_decl;
        {
            lu_init_barrier_decl << "nnfusion::cpu::Notification init_barrier;\n";
        }

        auto& init_body = projgen->lup_init->unit_vec;
        LanguageUnit_p init_barrier_notify =
            std::make_shared<LanguageUnit>("init_barrier_notify", "init_barrier.Notify();\n");
        init_body.push_back(init_barrier_notify);
        LanguageUnit_p init_barrier_wait =
            std::make_shared<LanguageUnit>("init_barrier_wait", "init_barrier.Wait();\n");
        body.insert(body.begin(), init_barrier_wait);
        // schedule thread pool
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

    if (global_required.count("header::reference_common") > 0)
    {
        projgen->lup_codegen->require(reference_common_header);
        reference_common_header->write_to = reference_common_header->symbol;
    }

    return true;
}