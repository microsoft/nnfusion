// Microsoft (c) 2019, NNFusion Team

#include <libgen.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cpu_codegenerator.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
// For reference kernels
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/reference/reference_common.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

namespace
{
    bool create_dir(std::string tar_path)
    {
        bool flag;
        int mkdir_status;
        struct stat s;
        int err = stat(tar_path.c_str(), &s);
        if (-1 == err)
        {
            mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (-1 == mkdir_status)
            {
                LOG(INFO) << "Error creating directory: " + tar_path;
                flag = false;
            }
            else
                flag = true;
        }
        else
        {
            flag = true;
        }
        return flag;
    }

    bool save_file(LanguageUnit_p lu)
    {
        std::ofstream out(lu->symbol);
        out << lu->get_code();
        out.close();
        return true;
    }
} // namespace

bool CpuCodeGenerator::setpwd()
{
    std::string working_dir = "./nnfusion_rt";
    std::string tar_path = working_dir + "/cpu_codegen/";
    create_dir(working_dir);
    create_dir(tar_path);
    int status = chdir(tar_path.c_str());
    return (bool)status;
}

bool CpuCodeGenerator::projgen()
{
    save_file(this->lu_cmakefile);
    save_file(this->lu_nnfusion_rt);
    save_file(this->lu_header);
    save_file(this->lu_main);
    return true;
}

bool CpuCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                           std::shared_ptr<TranslationUnit> tu)
{
    setpwd();

    auto& allocator_list = MemoryAllocatorFactory::get_allocator_list();
    auto async_manager = AsyncManagerFactory::get_async_manager(GENERIC_CPU);

    this->lu_cmakefile = LanguageUnit_p(new LanguageUnit("CMakeLists.txt"));
    this->lu_nnfusion_rt = LanguageUnit_p(new LanguageUnit("nnfusion_rt.cpp"));
    this->lu_header = LanguageUnit_p(new LanguageUnit("nnfusion_rt.h"));
    this->lu_main = LanguageUnit_p(new LanguageUnit("main_test.cpp"));

    LanguageUnit& lu_include = *this->lu_header;

    LanguageUnit lu_kernel_entry("KERNEL_ENTRY"); //func call for main()
    LanguageUnit lu_kernel_entry_header("KERNEL_ENTRY_HEADER");
    LanguageUnit lu_main_init("main_init");         //cuda_init()
    LanguageUnit lu_mem_plan_init("mem_plan_init"); // memory_pool planning in cuda_init()
    LanguageUnit lu_main_free("naive_free");        //cuda_free()
    LanguageUnit lu_thread_func_call("THREAD_FUNCTION_CALL");

    bool rc = true;

    std::vector<shared_ptr<KernelEmitter>> kernels;
    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        for (auto ins : *iterator)
        {
            string op_name = ins->getGNode()->get_op_type();
            if (!(*ins)["Async_info"].is_valid())
            {
                CHECK_FAIL() << "Async info should be be assigned before this pass:"
                             << ins->getGNode()->get_name();
            }
            shared_ptr<const KernelRegistration> kernel_reg = nullptr;

            std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
                KernelRegistry::Global()->FindKernelRegistrations(op_name, GENERIC_CPU, DT_FLOAT);

            shared_ptr<KernelContext> ctx(new KernelContext(ins->getGNode()));
            bool has_valid_kernel = false;
            if (kernel_regs.size() > 0)
            {
                for (auto kernel_reg : kernel_regs)
                {
                    auto kernel = kernel_reg->m_factory(ctx);
                    if (kernel->get_or_emit_source())
                    {
                        kernels.push_back(kernel);
                        has_valid_kernel = true;
                        break;
                    }
                }
            }

            if (kernel_regs.size() == 0 || !has_valid_kernel)
            {
                kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                    "AnyOP", GENERIC_CPU, DT_FLOAT);
                CHECK(kernel_reg != nullptr) << "AnyOp Kernel not found, op=" << op_name;
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();
                kernels.push_back(kernel);
            }
        }
    }

    LOG(INFO) << "Start dump whole source file...\n";
    // Code Gen
    LanguageUnit& lu = *this->lu_nnfusion_rt;
    lu << "// Microsoft (c) 2019, MSRA/NNFUSION Team\n";

    // Collect Requirement
    unordered_set<string> global_required;
    LanguageUnit re("REQUIREMENT");
    {
        re.require(header::assert);
        re.require(header::stdexcept);
        re.require(header::sstream);
        re.require(header::fstream);
        re.require(header::thread);
        if (async_manager->num_event() > 0)
            re.require(header::barrier);
        //re.require(declaration::typedef_int);

        for (auto kernel : kernels)
        {
            if (!(kernel->is_emitted()))
            {
                return false;
            }

            for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
            {
                re.require(it.second);
                global_required.insert(it.second->symbol);
            }
        }
    }

    lu << "#include \"nnfusion_rt.h\"\n\n";
    // std::unordered_map<string, vector<shared_ptr<KernelEmitter>>> stream_kernels;
    // Collect Function Definition
    {
        unordered_set<string> declared;
        LanguageUnit def("FUNCTIONS");
        for (auto kernel : kernels)
        {
            FunctionUnit_p fu = kernel->get_or_emit_source();
            for (auto& it : fu->body_unit->local_symbol)
            {
                if (it.second != fu->dep_unit)
                {
                    re.require(it.second);
                    global_required.insert(it.second->symbol);
                }
            }
            def << fu->comment_unit->get_code();
            if (declared.count(fu->body_unit->symbol) == 0)
            {
                def << fu->get_specialized_signature() << "\n";
                def.block_begin();
                def << fu->body_unit->get_code() << "\n";
                def.block_end();
                declared.insert(fu->body_unit->symbol);
            }
            else
            {
                def << "// Function declared:" << fu->body_unit->symbol << "\n\n";
            }
        }

        //Write Dependency
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("header::") != string::npos)
                lu << it.second->get_code();
        lu << "#include<cstring>\n";
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("macro::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("declaration::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        // stream and event declaration
        if (async_manager->num_stream() > 0)
            lu << async_manager->emit_stream_decl()->get_code();
        if (async_manager->num_event() > 0)
            lu << async_manager->emit_event_decl()->get_code();
        /*
        {
            //hard coded order
            if(re.local_symbol.find("cpu_reference_sum") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_product") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_max") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_min") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_softmax") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_broadcast") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_argmax") != re.local_symbol.end() ||
            re.local_symbol.find("cpu_reference_argmin") != re.local_symbol.end()
            )
            {
                re.local_symbol.erase("cpu_reference_reduce");
                //
            }
        }
        */
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("cpu_reference_") != string::npos)
                lu << it.second->get_code();
        lu << "\n";

        //Write Code
        lu << def.collect_code() << "\n";

        if (re.local_symbol.find("header::reference_common") != re.local_symbol.end())
        {
            save_file(reference_common_header);
        }
        if (async_manager->num_event() > 0)
            save_file(barrier_header);
    }

    // Generate caller function body
    {
        unordered_set<string> allocated;
        std::string kernel_entry_params;
        std::string kernel_entry_args;
        lu_kernel_entry << "extern \"C\" int kernel_entry(";
        // Add param
        {
            vector<string> params;
            vector<string> args;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
                args.push_back(tv->get_name());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
                args.push_back(tv->get_name());
            }
            kernel_entry_params = join(params, ", ");
            kernel_entry_args = join(args, ", ");

            lu_kernel_entry << kernel_entry_params;
        }

        lu_kernel_entry << ")";
        lu_kernel_entry_header << lu_kernel_entry.get_code();
        lu_kernel_entry << "\n";
        lu_kernel_entry.block_begin();

        // reset event/notification
        lu_kernel_entry << async_manager->emit_event_reset()->get_code();

        //Planning
        for (const auto& allocator : allocator_list)
        {
            lu_mem_plan_init << allocator.second->emit_memory_init()->get_code();
            lu_main_init << allocator.second->emit_memory_alloc()->get_code();
        }

        //Function Call
        {
            if (global_required.count("declaration::eigen_global_thread_pool") > 0)
            {
                lu_main_init << "int thread_count = std::thread::hardware_concurrency() >> 1;\n"
                             << "global_thread_pool = new Eigen::ThreadPool(thread_count? "
                                "thread_count : 1);\n";
            }
            if (global_required.count("declaration::eigen_global_thread_pool_device") > 0)
            {
                lu_main_init << "global_thread_pool_device = new "
                                "Eigen::ThreadPoolDevice(global_thread_pool, "
                                "global_thread_pool->NumThreads());";
            }
            if (global_required.count("declaration::mlas_global_thread_pool") > 0)
            {
                lu_main_init << "int thread_count = std::thread::hardware_concurrency() >> 1;\n"
                             << "mlas_global_thread_pool = new MLAS_THREADPOOL(\"mlas\", "
                                "thread_count ? thread_count : 1);\n";
            }

            std::unordered_map<string, vector<shared_ptr<KernelEmitter>>> stream_kernels_entry;
            for (auto kernel : kernels)
            {
                auto gnode = kernel->m_context->gnode;
                auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                auto stream = async_info.execution_stream;

                FunctionUnit_p fu = kernel->get_or_emit_source();
                //std::string read_const = fu->call_unit->get_code();
                std::string func_name = fu->name_unit->get_code();
                // emit function calls in cpu_init()
                //if (read_const.compare(0, 10, "read_const") == 0)
                if (func_name.compare(0, 9, "Constant_") == 0)
                {
                    CHECK(stream->is_default_stream()) << "Kernel function calls in cpu_init() "
                                                          "should use default/main stream/thread.";

                    // if (!async_info.wait_events.empty())
                    // {
                    //     for (auto event : async_info.wait_events)
                    //     {
                    //         lu_main_init << async_manager->emit_event_wait(async_info.execution_stream, event)->get_code();
                    //     }
                    // }
                    lu_main_init << fu->name_unit->get_code();
                    lu_main_init << fu->call_unit->get_code();

                    // if (async_info.record_event != nullptr)
                    // {
                    //     lu_main_init << async_manager->emit_event_record(async_info.record_event)->get_code();
                    // }
                }
                // organize kernels according to their streams/threads
                else
                {
                    std::string stream_name =
                        stream->is_default_stream() ? "default" : stream->get_name();
                    stream_kernels_entry[stream_name].push_back(kernel);
                }
            }
            // emit function calls in kernel_entry()
            for (auto& sk : stream_kernels_entry)
            {
                auto& stream_name = sk.first;
                if (stream_name != "default")
                {
                    // add thread_calls definition
                    lu_thread_func_call << "extern \"C\" void " << stream_name << "_Call(";
                    lu_thread_func_call << kernel_entry_params << ")\n";
                    lu_thread_func_call.block_begin();
                    for (auto kernel : sk.second)
                    {
                        auto gnode = kernel->m_context->gnode;
                        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();

                        if (!async_info.wait_events.empty())
                        {
                            for (auto event : async_info.wait_events)
                            {
                                lu_thread_func_call
                                    << async_manager
                                           ->emit_event_wait(async_info.execution_stream, event)
                                           ->get_code();
                            }
                        }

                        FunctionUnit_p fu = kernel->get_or_emit_source();
                        lu_thread_func_call << fu->name_unit->get_code();
                        lu_thread_func_call << fu->call_unit->get_code();

                        if (async_info.record_event != nullptr)
                        {
                            lu_thread_func_call
                                << async_manager->emit_event_record(async_info.record_event)
                                       ->get_code();
                        }
                    }
                    lu_thread_func_call.block_end();
                    // add function call to kernel entry
                    lu_kernel_entry << "std::thread " << stream_name << "(";
                    lu_kernel_entry << stream_name << "_Call, ";
                    lu_kernel_entry << kernel_entry_args << ");\n";
                }
                else
                {
                    for (auto kernel : sk.second)
                    {
                        FunctionUnit_p fu = kernel->get_or_emit_source();

                        {
                            auto gnode = kernel->m_context->gnode;
                            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                            if (!async_info.wait_events.empty())
                            {
                                for (auto event : async_info.wait_events)
                                {
                                    lu_kernel_entry
                                        << async_manager
                                               ->emit_event_wait(async_info.execution_stream, event)
                                               ->get_code();
                                }
                            }
                            lu_kernel_entry << fu->name_unit->get_code();
                            lu_kernel_entry << fu->call_unit->get_code();
                            if (async_info.record_event != nullptr)
                            {
                                lu_kernel_entry
                                    << async_manager->emit_event_record(async_info.record_event)
                                           ->get_code();
                            }
                        }
                    }
                }
            }
        }

        for (const auto& allocator : allocator_list)
        {
            lu_main_free << allocator.second->emit_memory_free()->get_code();
        }
        if (global_required.count("declaration::eigen_global_thread_pool") > 0)
        {
            lu_main_free << "free(global_thread_pool);\n";
        }
        if (global_required.count("declaration::eigen_global_thread_pool_device") > 0)
        {
            lu_main_free << "free(global_thread_pool_device);\n";
        }
        if (global_required.count("declaration::mlas_global_thread_pool") > 0)
        {
            lu_main_free << "free(mlas_global_thread_pool);\n";
        }

        // emit cpu stream/threads joining
        if (async_manager->num_stream() > 0)
            lu_kernel_entry << async_manager->emit_stream_join()->get_code();
        lu_kernel_entry << "\nreturn 0;\n";
        lu_kernel_entry.block_end();
    }

    lu << "\n";
    {
        lu << lu_mem_plan_init.get_code();
        lu << "\nextern \"C\" void cpu_init()";
        lu.block_begin();
        {
            lu << lu_main_init.get_code();
        }
        lu.block_end();
        lu << "\n";

        lu << "extern \"C\" void cpu_free()";
        lu.block_begin();
        {
            lu << lu_main_free.get_code();
        }

        lu.block_end();
    }
    lu << "\n";
    lu << lu_thread_func_call.get_code() << "\n";
    lu << lu_kernel_entry.get_code() << "\n\n";

    // generate main() function
    std::string function_include =
        R"(
#include <chrono>
#include <stdio.h>      
#include <stdlib.h>
#include "nnfusion_rt.h"
)";
    LanguageUnit& lu_main = *this->lu_main;
    {
        lu_main << function_include << "\n";
        lu_main << header::stdexcept->get_code();
        lu_main << "#include <sstream>\n";
        lu_main << "\n";
        lu_main << "using Clock = std::chrono::high_resolution_clock;\n\n";

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
                        << tensor.get_tensor_layout()->get_size() << " * "
                        << tensor.get_element_type().size() << ");\n";
            }

            lu_main << "\n//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                //malloc host output arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host = (" << tensor.get_element_type().c_type_string() << "*)"
                        << "malloc( sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << ");\n ";
            }

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name() + "_host");
            }
            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back(tv->get_name() + "_host");
            }

            lu_main << "\n//time measurement\n";
            lu_main << "auto t_start = Clock::now();\n\n";
            lu_main << "//kernel call\n";
            lu_main << "int steps = 100;\n";
            lu_main << "for(int i_=0; i_<steps; i_++)\n";
            lu_main.block_begin();
            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
            lu_main.block_end();

            lu_main << "\n//time measurement\n";
            lu_main << "auto t_end = Clock::now();\n";
            lu_main << "std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;\n\n";
            lu_main << "printf(\"function execution time: %f ms\\n\", fp_ms.count()/steps);\n";
            lu_main << "\n//free context\n";
            lu_main << "cpu_free();\n";
        }
        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "free(" << tensor.get_name() << "_host);\n";
        }
        //free host output args
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "free(" << tensor.get_name() << "_host);\n";
        }

        lu_main << "\nreturn 0;\n";
        lu_main.block_end();
    }

    //generate include header file
    lu_include << "// Microsoft (c) 2019\n";
    lu_include << "#pragma once\n";
    lu_include << declaration::typedef_int->get_code() << "\n";
    lu_include << lu_kernel_entry_header.get_code() << ";\n";
    lu_include << "extern \"C\" void cpu_init();\n";
    lu_include << "extern \"C\" void cpu_free();\n";

    //generate CMakeList.txt
    LanguageUnit& lu_cmake = *this->lu_cmakefile;
    //lu_cmake << generate_cmakelists();
    lu_cmake << "project(main_test)\n"
             << "cmake_minimum_required(VERSION 3.5)\n\n";

    if (global_required.count("declaration::eigen_global_thread_pool_device") > 0 ||
        global_required.count("header::eigen_utils") > 0 ||
        global_required.count("header::eigen_tensor") > 0 ||
        global_required.count("header::mlas") > 0)
    {
        lu_cmake << "# need to specify the correct path of eigen\n"
                 << "set(EIGEN_DIR \"/usr/include/eigen3\")\n"
                 << "include_directories(${EIGEN_DIR})\n\n";
    }

    if (global_required.count("header::cblas") > 0)
    {
        lu_cmake << R"(
set(NNFUSION_THIRDPARTY_FOLDER ~/repo/Thirdparty)
# include(mkldnn.cmake)
set(MKL_LIBS libiomp5.so libmklml_intel.so)
set(MKL_ROOT ${NNFUSION_THIRDPARTY_FOLDER}/mkl/mkl_lnx)
add_library(libmkl INTERFACE)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${MKL_ROOT}/lib/${LIB})
endforeach()
        )"
                 << "\n";
    }

    lu_cmake << "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -std=gnu++11\")\n"
             << "add_library(nnfusion_cpu_rt nnfusion_rt.cpp)\n";
    if (global_required.count("header::cblas") > 0)
    {
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt pthread libmkl)\n\n";
    }
    else
    {
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt pthread)\n\n";
    }

    if (global_required.count("header::mlas") > 0)
    {
        lu_cmake << "find_package(Threads REQUIRED)\n";
        lu_cmake << "include(mlas/mlas.cmake)\n";
        lu_cmake << "target_link_libraries(nnfusion_cpu_rt Threads::Threads mlas)\n\n";

        char exe_path[PATH_MAX];
        size_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
        const char* path;
        if (count != -1)
        {
            path = dirname(exe_path);
        }
        else
        {
            throw std::runtime_error("Failed to get the directory of executable file.\n");
        }
        std::string mlas_path = std::string(path) + std::string("/mlas");
        std::string cmd = std::string("cp -R ") + mlas_path + std::string(" .");
        if (0 != system(cmd.c_str()))
        {
            throw std::runtime_error("Failed to copy mlas source files.\n");
        }
    }

    lu_cmake << "target_compile_options(nnfusion_cpu_rt PRIVATE \"-fPIC\")\n"
             << "add_executable(main_test main_test.cpp)\n"
             << "target_link_libraries(main_test nnfusion_cpu_rt)\n";

    if (global_required.count("header::mlas") > 0)
    {
        lu_cmake << "target_link_libraries(main_test mlas)\n";
    }

    lu_cmake << R"(
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
else()
add_custom_command(
    TARGET nnfusion_cpu_rt
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()
)";

    projgen();

    // change to working directory
    int status = chdir("../../");
    return rc;
}
