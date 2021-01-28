// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#include <cstdio>
#include <libgen.h>
#include <limits.h>

#include "cpu_runtime.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cpu/reference/reference_common.hpp"

using namespace nnfusion::profiler;
using namespace nnfusion::kernels;

bool ReferenceRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;
    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code() + ".cpp");
    writer << boilerplate::MIT1->get_code();

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::fstream);
    re->require(header::thread);
    re->require(declaration::typedef_int);

    // Write Dependency
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
        {
            if (it.second->symbol == "header::reference_common")
            {
                writer << "// Unfolded reference_common.h begins\n";
                writer << reference_common_header->get_code();
                writer << "using namespace reference_common;\n";
                writer << "// Unfolded reference_common.h ends\n";
            }
            else
                writer << it.second->get_code();
        }
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();

    writer << "#include <chrono>\n#include <ctime>\n#include <ratio>\n#include <cmath>\n#include "
              "<numeric>\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("cpu_reference_") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    //Write Code

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << fu->body_unit->get_code() << "\n";
    writer.block_end();

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;
    auto& temp = ke->kernel->m_context->tensors;

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i]->get_element_type().c_type_string() << "* " << arg[i]->get_name() << ", ";
    }
    if (!arg.empty())
    {
        writer << arg.back()->get_element_type().c_type_string() << "* " << arg.back()->get_name();
        if (!out.empty())
            writer << ", ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i]->get_element_type().c_type_string() << "* " << out[i]->get_name() << ", ";
    }
    if (!out.empty())
    {
        writer << out.back()->get_element_type().c_type_string() << "* " << out.back()->get_name();
    }
    writer << ")\n";

    auto tensor_declare = [](const shared_ptr<nnfusion::descriptor::Tensor>& t) -> std::string {
        return t->get_element_type().c_type_string() + "* " + t->get_name() + ";\n";
    };

    auto tensor_alloc_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << tensor->get_name() << " = new " << tensor->get_element_type().c_type_string() << "["
          << tensor->size(false) << "];\n";
        return s.str();
    };

    auto tensor_free_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << "delete[] " << tensor->get_name() << ";\n";
        return s.str();
    };

    auto tensor_memset_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << "memset(" << tensor->get_name() << tensor->get_memset_value() << ", "
          << tensor->size(false) << " * " << tensor->get_element_type().size() << ");\n";
        return s.str();
    };

    fu = ke->kernel->get_or_emit_source(true);
    writer.block_begin();
    {
        for (size_t i = 0; i < temp.size(); i++)
        {
            writer << tensor_declare(temp[i]);
            writer << tensor_alloc_host(temp[i]);
            if (temp[i]->is_memset())
            {
                writer << tensor_memset_host(temp[i]);
            }
        }

        writer << "std::chrono::high_resolution_clock::time_point t1,t2;\n";
        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times
                   << ") t1 = std::chrono::high_resolution_clock::now();\n";
            writer << fu->name_unit->get_code() << fu->call_unit->get_code();
        }
        writer.block_end();
        writer << "t2 = std::chrono::high_resolution_clock::now();\n";
        writer << "std::chrono::duration<double> time_span = "
                  "std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);\n";
        writer << "double milliseconds = time_span.count();\n";
        writer << "return milliseconds/" << ke->runtime_times << ";\n";

        for (size_t i = 0; i < temp.size(); i++)
            writer << tensor_free_host(temp[i]);
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "]";
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    return true;
}

bool ReferenceRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = filename + ".cpp";
    // ofstream source_file(ke->working_dir + "/" + ke->source_code->symbol);
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    int ret = system(("gcc\t-fPIC\t-shared\t-std=c++11\t" + srcname + "\t-o\t" + objname).c_str());
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

double ReferenceRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    // Replacing Existed Kernel with Reference Kenel
    auto& gnode = ke->kernel->m_context->gnode;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), GENERIC_CPU, element::f32);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    bool has_valid_kernel = false;
    for (auto kernel_reg : kernel_regs)
    {
        if (kernel_reg->m_tag != "reference")
            continue;
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            NNFUSION_LOG(INFO) << "Replacing with reference kenel.";
            // Replacing the kernel;
            ke->kernel = kernel;
        }
    }

    if (!has_valid_kernel)
        return -1.0;
    if (codegen(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

ReferenceRuntime::Pointer ReferenceRuntime::Runtime()
{
    static ReferenceRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<ReferenceRuntime>();
    return predefined;
}

bool CPUDefaultRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;

    // setpwd
    std::string working_dir = "./cpu_profiler/";
    nnfusion::codegen::create_folder(working_dir);
    int status = chdir(working_dir.c_str());
    NNFUSION_CHECK(status == 0);

    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code());
    writer << boilerplate::MIT1->get_code();

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::fstream);
    re->require(header::thread);
    re->require(declaration::typedef_int);
    if (ke->kernel->is_parallelism())
    {
        re->require(header::threadpool);
    }
    for (auto& it : re->local_symbol)
        global_required.insert(it.second->symbol);

    // Write Dependency
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
        {
            if (it.second->symbol == "header::reference_common")
            {
                writer << "// Unfolded reference_common.h begins\n";
                writer << reference_common_header->get_code();
                writer << "using namespace reference_common;\n";
                writer << "// Unfolded reference_common.h ends\n";
            }
            else
                writer << it.second->get_code();
        }
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();

    writer << "#include <chrono>\n#include <ctime>\n#include <ratio>\n#include <cmath>\n#include "
              "<numeric>\n#include<cstring>\nusing namespace std;\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("cpu_reference_") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    //Write Code

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << fu->body_unit->get_code() << "\n";
    writer.block_end();

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;
    auto& temp = ke->kernel->m_context->tensors;

    // Dedupe Tensors
    std::map<std::string, size_t> arg_cnt;
    std::map<std::string, size_t> out_cnt;
    std::map<std::string, size_t> temp_cnt;
    std::vector<std::string> arg_name_dedupe;
    std::vector<std::string> out_name_dedupe;
    std::vector<std::string> temp_name_dedupe;
    for (size_t i = 0; i < arg.size(); i++)
    {
        auto name = arg[i]->get_name();
        if (arg_cnt.find(name) == arg_cnt.end())
        {
            arg_cnt[name] = 1;
            arg_name_dedupe.push_back(name);
        }
        else
        {
            arg_name_dedupe.push_back(name + std::to_string(arg_cnt[name]));
            arg_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < out.size(); i++)
    {
        auto name = out[i]->get_name();
        if (out_cnt.find(name) == out_cnt.end())
        {
            out_cnt[name] = 1;
            out_name_dedupe.push_back(name);
        }
        else
        {
            out_name_dedupe.push_back(name + std::to_string(out_cnt[name]));
            out_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < temp.size(); i++)
    {
        auto name = temp[i]->get_name();
        if (temp_cnt.find(name) == temp_cnt.end())
        {
            temp_cnt[name] = 1;
            temp_name_dedupe.push_back(name);
        }
        else
        {
            temp_name_dedupe.push_back(name + std::to_string(temp_cnt[name]));
            temp_cnt[name] += 1;
        }
    }

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    auto idx = fu->name_unit->get_code().find("Result");

    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i]->get_element_type().c_type_string() << "* " << arg_name_dedupe[i] << ", ";
    }
    if (!arg.empty())
    {
        writer << arg.back()->get_element_type().c_type_string() << "* " << arg_name_dedupe.back();
        if (!out.empty())
            writer << ", ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        if (idx != string::npos)
            writer << out[i]->get_element_type().c_type_string() << "** " << out_name_dedupe[i]
                   << ", ";
        else
            writer << out[i]->get_element_type().c_type_string() << "* " << out_name_dedupe[i]
                   << ", ";
    }
    if (!out.empty())
    {
        if (idx != string::npos)
            writer << out.back()->get_element_type().c_type_string() << "** "
                   << out_name_dedupe.back();
        else
            writer << out.back()->get_element_type().c_type_string() << "* "
                   << out_name_dedupe.back();
    }
    writer << ")\n";

    auto tensor_declare = [](const shared_ptr<nnfusion::descriptor::Tensor>& t) -> std::string {
        return t->get_element_type().c_type_string() + "* " + t->get_name() + ";\n";
    };

    auto tensor_alloc_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << tensor->get_name() << " = new " << tensor->get_element_type().c_type_string() << "["
          << tensor->size(false) << "];\n";
        return s.str();
    };

    auto tensor_free_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << "delete[] " << tensor->get_name() << ";\n";
        return s.str();
    };

    auto tensor_memset_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
        stringstream s;
        s << "memset(" << tensor->get_name() << tensor->get_memset_value() << ", "
          << tensor->size(false) << " * " << tensor->get_element_type().size() << ");\n";
        return s.str();
    };

    fu = ke->kernel->get_or_emit_source(true);
    writer.block_begin();
    {
        if (global_required.count("header::eigen_spatial_convolution") > 0)
        {
            writer << "setenv(\"OMP_NUM_THREADS\", \"1\", true);\n";
        }

        if (ke->kernel->is_parallelism())
        {
            writer << declaration::worker_thread_pool->get_code() << "\n";
            writer << "worker_thread_pool = new concurrency::NumaAwareThreadPool(1, 1);\n";
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                writer << tensor_alloc_host(tensor);
                if (tensor->is_memset())
                {
                    writer << tensor_memset_host(tensor);
                }
            }
        }

        writer << "std::chrono::high_resolution_clock::time_point t1,t2;\n";
        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times
                   << ") t1 = std::chrono::high_resolution_clock::now();\n";
            writer << fu->name_unit->get_code();
            std::string call_str = fu->call_unit->get_code();
            if (ke->kernel->is_parallelism())
            {
                std::string threadpool_param = "worker_thread_pool->GetRawThreadPool(), ";
                call_str.insert(1, threadpool_param);
            }
            writer << call_str;
        }
        writer.block_end();
        writer << "t2 = std::chrono::high_resolution_clock::now();\n";
        writer
            << "std::chrono::duration<double, std::micro> time_span = "
               "std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t2 - t1);\n";
        writer << "double milliseconds = time_span.count();\n";

        // free tensors
        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_free_host(tensor);
            }
        }

        if (ke->kernel->is_parallelism())
        {
            writer << "delete worker_thread_pool;\n";
        }

        writer << "return milliseconds/" << ke->runtime_times << ";\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            if (idx != string::npos)
            {
                writer << "(" << type << "**)" << (out.size() == 0 ? "args" : "outputs") << "["
                       << i - (i >= arg.size() ? arg.size() : 0) << "]";
            }
            else
            {
                writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                       << i - (i >= arg.size() ? arg.size() : 0) << "]";
            }
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    // save src file
    string filename = ke->source_code->get_symbol();
    if (filename.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(filename);
        filename = "compressed_src_" + std::to_string(hashcode);
    }

    string srcname = filename + ".cpp";
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    status = chdir("../");
    NNFUSION_CHECK(status == 0);
    return true;
}

bool CPUDefaultRuntime::cmake_codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->cmake_code != nullptr)
        return true;

    // setpwd
    std::string working_dir = "./cpu_profiler/";
    nnfusion::codegen::create_folder(working_dir);
    int status = chdir(working_dir.c_str());
    NNFUSION_CHECK(status == 0);

    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit lu_cmake("CMakeLists.txt");
    auto re = fu->dep_unit;
    lu_cmake << boilerplate::MIT2->get_code();
    lu_cmake << R"(
project(cpu_profiler)
cmake_minimum_required(VERSION 3.5)

SET(SOURCE_FILE "" CACHE STRING "cpu kernel profiler source files")
if(EXISTS "${SOURCE_FILE}")
else()
message(SEND_ERROR "SOURCE_FILE not exists." )
endif()

SET(TARGET_NAME "" CACHE STRING "cpu kernel profiler target name")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++11 -O3 -march=native -pthread")
add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})
    )"
             << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol == "header::cblas")
        {
            lu_cmake << R"(
set(NNFUSION_THIRDPARTY_FOLDER "~/repo/Thirdparty" CACHE STRING "NNFusion Thirdpary libraries folder location")
if(EXISTS "${NNFUSION_THIRDPARTY_FOLDER}")
else()
message(SEND_ERROR "NNFUSION_THIRDPARTY_FOLDER not exists." )
endif()
# include(mkldnn.cmake)
set(MKL_LIBS libiomp5.so libmklml_intel.so)
set(MKL_ROOT ${NNFUSION_THIRDPARTY_FOLDER}/mkl/mkl_lnx)
add_library(libmkl INTERFACE)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${MKL_ROOT}/lib/${LIB})
endforeach()

target_link_libraries(${TARGET_NAME} pthread libmkl)
            )"
                     << "\n";
        }

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

    struct stat s;

    lu_cmake << "find_package(Threads REQUIRED)\n";
    lu_cmake << "target_link_libraries(${TARGET_NAME} Threads::Threads)\n\n";

    if (ke->kernel->is_parallelism())
    {
        // Prepare eigen submodule.
        if (stat("./eigen", &s) != 0)
        {
            std::string eigen_path = std::string(path) + std::string("/eigen");
            std::string cmd = std::string("cp -R ") + eigen_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy eigen source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET eigen)\n";
        lu_cmake << "include(eigen/eigen.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} eigen)\n\n";

        // Prepare threadpool submodule.
        if (stat("./threadpool", &s) != 0)
        {
            std::string threadpool_path = std::string(path) + std::string("/threadpool");
            std::string cmd = std::string("cp -R ") + threadpool_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy threadpool source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET threadpool)\n";
        lu_cmake << "include(threadpool/threadpool.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} threadpool)\n\n";
    }

    if (global_required.count("header::mlas") > 0)
    {
        // Prepare mlas submodule.
        if (stat("./mlas", &s) != 0)
        {
            std::string mlas_path = std::string(path) + std::string("/mlas");
            std::string cmd = std::string("cp -R ") + mlas_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy mlas source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET mlas)\n";
        lu_cmake << "include(mlas/mlas.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} mlas)\n\n";
    }

    ke->cmake_code = make_shared<LanguageUnit>(move(lu_cmake));

    // save cmake file
    string cmakename = "CMakeLists.txt";
    ofstream cmake_file(cmakename);
    cmake_file << ke->cmake_code->get_code();
    cmake_file.close();

    status = chdir("../");
    NNFUSION_CHECK(status == 0);
    return true;
}

bool CPUDefaultRuntime::general_cmake_codegen()
{
    // setpwd
    std::string working_dir = "./cpu_profiler/";
    nnfusion::codegen::create_folder(working_dir);
    int status = chdir(working_dir.c_str());
    NNFUSION_CHECK(status == 0);

    LanguageUnit lu_cmake("CMakeLists.txt");
    lu_cmake << boilerplate::MIT2->get_code();
    lu_cmake << R"(
project(cpu_profiler)
cmake_minimum_required(VERSION 3.5)

SET(TARGET_NAME "" CACHE STRING "cpu kernel profiler target name")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++11 -O3 -march=native -pthread")

file(GLOB SOURCE_FILE *.cpp)

add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})
    )"
             << "\n";

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

    struct stat s;

    lu_cmake << "find_package(Threads REQUIRED)\n";
    lu_cmake << "target_link_libraries(${TARGET_NAME} Threads::Threads)\n\n";

    // if (ke->kernel->is_parallelism())
    {
        // Prepare eigen submodule.
        if (stat("./eigen", &s) != 0)
        {
            std::string eigen_path = std::string(path) + std::string("/eigen");
            std::string cmd = std::string("cp -R ") + eigen_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy eigen source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET eigen)\n";
        lu_cmake << "include(eigen/eigen.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} eigen)\n\n";

        // Prepare threadpool submodule.
        if (stat("./threadpool", &s) != 0)
        {
            std::string threadpool_path = std::string(path) + std::string("/threadpool");
            std::string cmd = std::string("cp -R ") + threadpool_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy threadpool source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET threadpool)\n";
        lu_cmake << "include(threadpool/threadpool.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} threadpool)\n\n";
    }

    // if (global_required.count("header::mlas") > 0)
    {
        // Prepare mlas submodule.
        if (stat("./mlas", &s) != 0)
        {
            std::string mlas_path = std::string(path) + std::string("/mlas");
            std::string cmd = std::string("cp -R ") + mlas_path + std::string(" .");
            if (0 != system(cmd.c_str()))
            {
                throw std::runtime_error("Failed to copy mlas source files.\n");
            }
        }
        lu_cmake << "if (NOT TARGET mlas)\n";
        lu_cmake << "include(mlas/mlas.cmake)\n";
        lu_cmake << "endif()\n";
        lu_cmake << "target_link_libraries(${TARGET_NAME} mlas)\n\n";
    }

    // save cmake file
    string cmakename = "CMakeLists.txt";
    ofstream cmake_file(cmakename);
    cmake_file << lu_cmake.get_code();
    cmake_file.close();

    status = chdir("../");
    NNFUSION_CHECK(status == 0);
    return true;
}

bool CPUDefaultRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    // setpwd
    std::string working_dir = "./cpu_profiler/";
    nnfusion::codegen::create_folder(working_dir);
    int status = chdir(working_dir.c_str());
    NNFUSION_CHECK(status == 0);

    // src file
    string filename = ke->source_code->get_symbol();
    if (filename.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(filename);
        filename = "compressed_src_" + std::to_string(hashcode);
    }

    string objname = std::string("lib") + filename + DLIB_SUFFIX;
    string srcname = filename + ".cpp";

    std::string cmd = std::string("cmake . -DSOURCE_FILE=") + srcname +
                      std::string(" -DTARGET_NAME=") + filename + string("&& make -j");
    int ret = system((cmd.c_str()));
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;

    status = chdir("../");
    NNFUSION_CHECK(status == 0);
    return true;
}

bool CPUDefaultRuntime::general_compile()
{
    // generate cmake file
    if (!general_cmake_codegen())
        return false;

    // setpwd
    std::string working_dir = "./cpu_profiler/";
    nnfusion::codegen::create_folder(working_dir);
    int status = chdir(working_dir.c_str());
    NNFUSION_CHECK(status == 0);

    string objname = std::string("libcpu_kernel_prof") + DLIB_SUFFIX;

    std::string cmd = std::string("cmake . -DTARGET_NAME=cpu_kernel_prof && make -j");
    int ret = system((cmd.c_str()));
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;

    status = chdir("../");
    NNFUSION_CHECK(status == 0);
    return true;
}

double CPUDefaultRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    global_required.clear();

    if (codegen(ke) == false)
    {
        NNFUSION_LOG(INFO) << "cpu kernel source file codegen fail.";
        return -1.0;
    }
    if (cmake_codegen(ke) == false)
    {
        NNFUSION_LOG(INFO) << "cpu kernel cmake file codegen fail.";
        return -1.0;
    }
    if (compile(ke) == false)
    {
        NNFUSION_LOG(INFO) << "cpu kernel compilation fail.";
        return -1.0;
    }
    if (ke->entry_point == nullptr)
    {
        NNFUSION_LOG(INFO) << "cpu kernel entry point fail.";
        return -1.0;
    }
    return ke->entry_point(input, output);
}

double
    CPUDefaultRuntime::sep_invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    // setpwd
    std::string working_dir = "./cpu_profiler/";
    // nnfusion::codegen::create_folder(working_dir);
    // int status = chdir(working_dir.c_str());
    // NNFUSION_CHECK(status == 0);

    std::string objname = working_dir + std::string("libcpu_kernel_prof") + DLIB_SUFFIX;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
    {
        NNFUSION_LOG(INFO) << "cpu kernel entry not found.";
        return -1.0;
    }
    ke->entry_point = (double (*)(void**, void**))entry;

    // status = chdir("../");
    // NNFUSION_CHECK(status == 0);

    if (ke->entry_point == nullptr)
    {
        NNFUSION_LOG(INFO) << "cpu kernel entry point fail.";
        return -1.0;
    }
    return ke->entry_point(input, output);
}

CPUDefaultRuntime::Pointer CPUDefaultRuntime::Runtime()
{
    static CPUDefaultRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<CPUDefaultRuntime>();
    return predefined;
}