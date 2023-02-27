// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "codegen_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "rocm_codegen_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

DECLARE_bool(fkernels_as_files);
DECLARE_int64(fkernels_files_number);
DECLARE_bool(ffunction_codegen);

void RocmCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    set_global_member(ctx, tu);

    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.cu";
    auto& copy_templates = projgen->lup_codegen->copy_templates;
    copy_templates.emplace_back("rocm_adapter/rocm_adapter.h", "./rocm_adapter.h");
    // copy_templates.emplace_back("rocm_adapter/fastgen_for_sliced_kernels.sh",
    //                             "./fastgen_for_sliced_kernels.sh");
    // NNFUSION_CHECK(0 == system("chmod a+x fastgen_for_sliced_kernels.sh"));
    copy_templates.emplace_back("image_tests/image_test.cpp", "./image_tests/image_test.cpp");
    copy_templates.emplace_back("image_tests/CMakeLists_rocm.txt", "./image_tests/CMakeLists.txt");

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

    if (superscaler_enable)
    {
        std::string superscaler_path = std::string(path) + std::string("/superscaler");
        copy_folder.push_back(superscaler_path);
    }
    copy_templates.emplace_back("image_tests/image_test.cpp", "./image_tests/image_test.cpp");
    copy_templates.emplace_back("image_tests/CMakeLists_rocm.txt", "./image_tests/CMakeLists.txt");

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        if (FLAGS_ffunction_codegen)
            lu_init_begin << "\nextern \"C\" void cuda_init(char* workspace)\n{\n";
        else
            lu_init_begin << "\nextern \"C\" void cuda_init()\n{\n";
        lu_init_begin << "// CUDA_SAFE_CALL(cudaDeviceReset());\n";
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
        lu_exit_begin << "extern \"C\" void cuda_free()\n{\n";
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        lu_exit_end << "}\n";
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
    projgen->lup_codegen->require(macro::TVM_PACK_VALUES);
    projgen->lup_codegen->require(codegen_device_type());
    projgen->lup_codegen->require(codegen_workspace_size(tu));

    return;
}

// todo: add flags for future.
void RocmCodegenPass::create_cmake_file(std::shared_ptr<InterpreterContext> ctx,
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
SET(TARGET_NAME "nnfusion_naive_rt" CACHE STRING "codegen target name")

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "-O2 -ffast-math -Wno-ignored-attributes -Wno-duplicate-decl-specifier")
)";
    if (FLAGS_fkernels_as_files)
    {
        lu << "\nfile(GLOB kernels kernels/*"
           << ".cpp"
           << ")\n";
        lu << "list(APPEND SRC ${kernels} shared"
           << ".cpp"
           << ")\n";
        lu << "include_directories(${CMAKE_SOURCE_DIR})\n\n";
    }
    lu << "add_library(${TARGET_NAME} SHARED ${SRC})\n";

    // Prepare submodule
    {
        // add rocm_lib
        lu << nnfusion::codegen::cmake::rocm_lib->get_code();

        if (superscaler_enable)
        {
            // add superscaler
            lu << nnfusion::codegen::cmake::superscaler_rocm->get_code();
        }
    }

    lu << R"(
add_executable(main_test main_test.cpp)
target_link_libraries(main_test ${TARGET_NAME})
)";
    return;
}

bool RocmCodegenPass::after_projgen()
{
    struct stat s;
    std::string constant_folder = get_current_dir_name() + std::string("/Constant");
    if (stat(constant_folder.c_str(), &s) == 0)
    {
        std::string cmd = std::string("cp -R ") + constant_folder + " " + m_codegen_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to copy constant files.\n");
        }
    }
    auto cd = get_current_dir_name();
    NNFUSION_CHECK(chdir(m_codegen_folder.c_str()) == 0);
    {
        auto hipify_exec =
            nnfusion::codegen::get_file_from_templates("rocm_adapter/hipify-nnfusion");
        // update for nnfusion_rt.cu
        NNFUSION_CHECK(
            0 == system((hipify_exec +
                         " nnfusion_rt.cu | grep -v 'include.*cublas_v2' | grep -v "
                         "'include.*cuda.h' | grep -v 'include.*cudnn' > nnfusion_rt.cpp && rm "
                         "nnfusion_rt.cu")
                            .c_str()));

        // update for shared.h and shared.cu
        if (projgen->need_shared_file())
        {
            NNFUSION_CHECK(0 == system("mv shared.h shared.h.old"));
            NNFUSION_CHECK(0 == system("mv shared.cu shared.cu.old"));

            NNFUSION_CHECK(0 ==
                           system((hipify_exec + " shared.h.old | grep -v 'include.*cublas_v2' | "
                                                 "grep -v 'include.*cudnn' > shared.h && rm "
                                                 "shared.h.old")
                                      .c_str()));

            NNFUSION_CHECK(0 == system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' shared.h"));

            NNFUSION_CHECK(
                0 == system((hipify_exec +
                             " shared.cu.old | grep -v 'include.*cublas_v2' | grep -v "
                             "'include.*cuda.h' | grep -v 'include.*cudnn' > shared.cpp && rm "
                             "shared.cu.old")
                                .c_str()));
        }

        // for rocm 3.5 compatibility purpose
        string rocm35cmd = "sed -i 's/extern *__shared__/__shared__/g' nnfusion_rt.cpp";
        NNFUSION_CHECK(0 == system((rocm35cmd).c_str())) << rocm35cmd;

        // update for main_test.cpp
        NNFUSION_CHECK(
            0 == system("sed -i 's/^.*include.*cuda_profiler_api.*$//g' main_test.cpp && sed -i "
                        "'s/cudaProfiler.*\\(.*\\)//g' main_test.cpp"));
        // update for nnfusion_rt.h
        NNFUSION_CHECK(0 ==
                       system("sed -i 's/<cuda\\.h>/\"rocm_adapter.h\"/g' nnfusion_rt.h && sed -i "
                              "'s/cuda_runtime\\.h/hip\\/hip_runtime.h/g' nnfusion_rt.h"));
        // Update for kernels
        if (FLAGS_fkernels_as_files)
        {
            auto list_dir = [](string path, string has_str) {
                vector<string> files;
                struct dirent* entry;
                DIR* dir = opendir(path.c_str());
                if (dir == NULL)
                {
                    return files;
                }

                while ((entry = readdir(dir)) != NULL)
                {
                    string tstr(entry->d_name);
                    if (tstr.find(has_str) < tstr.length() && tstr.length() > 3)
                        files.push_back(path + "/" + tstr);
                }
                closedir(dir);
                return files;
            };

            auto kernels = list_dir("kernels", ".cu");
            for (auto kernel : kernels)
            {
                auto raw_kernel = kernel;
                kernel.replace(kernel.find(".cu"), 3, ".cpp");
                string grepv = hipify_exec + " " + raw_kernel +
                               " | grep -v 'include.*cublas_v2' | grep -v 'include.*cuda.h' | "
                               "grep -v 'include.*cudnn' > " +
                               kernel + " && rm " + raw_kernel;
                NNFUSION_CHECK(0 == system((grepv).c_str())) << grepv;
                string hipcmd =
                    "sed -i 's/#include <hip\\/hip_runtime.h>/#include "
                    "\"..\\/rocm_adapter.h\"\\n#include "
                    "<hip\\/hip_runtime.h>/g' " +
                    kernel;
                NNFUSION_CHECK(0 == system((hipcmd).c_str())) << hipcmd;
                hipcmd = "sed -i 's/extern *__shared__/__shared__/g' " + kernel;
                NNFUSION_CHECK(0 == system((hipcmd).c_str())) << hipcmd;
            }
        }
        // // update for rocm_adapter.h
        // nnfusion::codegen::copy_file_from_templates("rocm_adapter/rocm_adapter.h", "./rocm_adapter.h");
        // // fast compile script for dynamic shared lib
        // nnfusion::codegen::copy_file_from_templates("rocm_adapter/fastgen_for_sliced_kernels.sh",
        //                                             "./fastgen_for_sliced_kernels.sh");
        // NNFUSION_CHECK(0 == system("chmod a+x fastgen_for_sliced_kernels.sh"));
    }
    NNFUSION_CHECK(chdir(cd) == 0);
    return true;
}
