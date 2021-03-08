// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_cpp_codegen_pass.hpp"
#include "codegenerator_helper.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_kernel_emitter.hpp"
#include "nnfusion/core/kernels/hlsl/hlsl_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

DECLARE_string(fdefault_device);
DECLARE_int32(fwarmup_step);
DECLARE_int32(frun_step);
DECLARE_bool(fcodegen_debug);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fcustomized_mem_imp);

void HLSLCPPCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                    std::shared_ptr<TranslationUnit> tu)
{
    set_global_member(ctx, tu);
    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "Main.cpp";

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

    std::string Direct3DWinNN_path =
        std::string(path) + std::string("/templates/dxcompute/Direct3DWinNN");
    copy_folder.push_back(Direct3DWinNN_path);
    std::string Direct3DXBoxNN_path =
        std::string(path) + std::string("/templates/dxcompute/Direct3DXBoxNN");
    copy_folder.push_back(Direct3DXBoxNN_path);

    projgen->lup_codegen->require(lup_main);
    lup_main->require(projgen->lup_init);
    lup_main->require(projgen->lup_exec);
    lup_main->require(projgen->lup_exit);

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        lu_init_begin << "\nvoid hlsl_init()\n{\n";
    }

    auto& lu_init_end = *(projgen->lup_init->end);
    {
        lu_init_end << "dxStreamSynchronize(0);\n";
        lu_init_end << "}\n\n";
    }
    //     {
    //         LanguageUnit_p lib_init = std::make_shared<LanguageUnit>("lib_init", R"(
    // libtestdll = LoadLibrary(L"antares_hlsl_v0.1_x64.dll");
    // if (!libtestdll)
    // {
    //     fprintf(stderr, "Cannot find antares_hlsl_v0.1_x64.dll !\n");
    //     exit(1);
    // }

    // dxInit = (int (*)(int flags))GetProcAddress(libtestdll, "dxInit");

    // dxStreamCreate = (void* (*)())GetProcAddress(libtestdll, "dxStreamCreate");
    // dxStreamDestroy = (int (*)(void* hStream))GetProcAddress(libtestdll, "dxStreamDestroy");
    // dxStreamSubmit = (int (*)(void* hStream))GetProcAddress(libtestdll, "dxStreamSubmit");
    // dxStreamSynchronize = (int (*)(void* hStream))GetProcAddress(libtestdll, "dxStreamSynchronize");

    // dxMemAlloc = (void* (*)(size_t bytes))GetProcAddress(libtestdll, "dxMemAlloc");
    // dxMemFree = (int (*)(void* dptr))GetProcAddress(libtestdll, "dxMemFree");
    // dxMemcpyHtoDAsync = (int (*)(void* dst, void* src, size_t bytes, void* hStream))GetProcAddress(libtestdll, "dxMemcpyHtoDAsync");
    // dxMemcpyDtoHAsync = (int (*)(void* dst, void* src, size_t bytes, void* hStream))GetProcAddress(libtestdll, "dxMemcpyDtoHAsync");

    // dxShaderLoad = (void* (*)(const char* src, int* num_inputs, int* num_outputs))GetProcAddress(libtestdll, "dxShaderLoad");
    // dxShaderUnload = (int (*)(void* hShader))GetProcAddress(libtestdll, "dxShaderUnload");
    // dxShaderGetProperty = (int (*)(void* hShader, int arg_index, size_t * num_elements, size_t * type_size, const char** dtype_name))GetProcAddress(libtestdll, "dxShaderGetProperty");
    // dxShaderLaunchAsync = (int (*)(void* hShader, void** buffers, void* hStream))GetProcAddress(libtestdll, "dxShaderLaunchAsync");

    // dxEventCreate = (void* (*)())GetProcAddress(libtestdll, "dxEventCreate");
    // dxEventRecord = (int (*)(void* hEvent, void* hStream))GetProcAddress(libtestdll, "dxEventRecord");
    // dxEventElapsedTime = (float (*)(void* hStart, void* hStop))GetProcAddress(libtestdll, "dxEventElapsedTime");
    // dxEventDestroy = (int (*)(void* hEvent))GetProcAddress(libtestdll, "dxEventDestroy");

    // if (!dxShaderLoad)
    // {
    //     std::cout << "no valid dxShaderLoad func" << std::endl;
    //     exit(1);
    // }
    //         )");

    //         LanguageUnit_p lib_free =
    //             std::make_shared<LanguageUnit>("lib_free", "FreeLibrary(libtestdll);\n");

    //         lib_init->require(declaration::antares_hlsl_dll_cpp);
    //         lib_free->require(declaration::antares_hlsl_dll_cpp);
    //         add_init_and_exit_pair(lib_init, lib_free);
    //     }

    auto& lu_exec_begin = *(projgen->lup_exec->begin);
    {
        std::string params = get_kernel_entry_paras(tu);
        lu_exec_begin << "\nint kernel_entry(" << params << ")\n{\n";
    }

    auto& lu_exec_end = *(projgen->lup_exec->end);
    {
        lu_exec_end << "return 0;\n";
        lu_exec_end << "}\n\n";
    }

    auto& lu_exit_begin = *(projgen->lup_exit->begin);
    {
        lu_exit_begin << "\nvoid hlsl_free()\n{\n";
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        lu_exit_end << "}\n\n";
    }

    auto& lu_main_begin = *(lup_main->begin);
    {
        lu_main_begin << "\nint main()\n{\n";
    }

    auto& lup_main_end = *(lup_main->end);
    {
        lup_main_end << "return 0;\n";
        lup_main_end << "}\n\n";
    }

    // add requirement
    projgen->lup_codegen->require(header::iostream);
    projgen->lup_codegen->require(header::sstream);
    projgen->lup_codegen->require(header::stdio);
    projgen->lup_codegen->require(header::windows);
    projgen->lup_codegen->require(header::D3D12APIWrapper);
    projgen->lup_codegen->require(macro::OutputDebugStringA);
    LanguageUnit_p num_inputs_outputs = std::make_shared<LanguageUnit>(
        "declaration::num_inputs_outputs", "int num_inputs, num_outputs;\n");
    projgen->lup_codegen->require(num_inputs_outputs);

    generate_main(ctx, tu);

    return;
}

bool HLSLCPPCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;
    auto sharder_pair = create_init_and_exit_pair<LanguageUnitwithVec, LanguageUnitwithVec>(
        "shader_load", "shader_unload");
    auto lup_shaders_load = sharder_pair.first;
    auto lup_shaders_unload = sharder_pair.second;
    int count = 0;
    auto pairs = collect_ins(ctx, tu);
    for (size_t i = 0; i < pairs.size(); i++)
    {
        auto& it = pairs[i];
        int pos = it.first.find(":");
        NNFUSION_CHECK(pos >= 0);
        std::string thread_name = it.first.substr(pos + 1);
        std::string main_block = it.first.substr(0, pos);

        auto lup_func_calls = get_kernel_func_calls(it.first + "_func_calls", nullptr);
        for (auto ins : it.second)
        {
            auto kernel = ins->getKernel();
            auto gnode = ins->getGNode();
            if (gnode->get_op_ptr()->is_parameter())
                continue;
            auto& async_info = (*ins)["Async_info"].as<AsyncExecutionInfo>();
            FunctionUnit_p fu = kernel->get_or_emit_source(true);
            string body_str = fu->body_unit->get_code();
            string func_name = fu->name_unit->get_code();
            string hShader_name = func_name + "_hShader";
            if (!body_str.empty())
            {
                if (kernel_func_defs.find(body_str) == kernel_func_defs.end())
                {
                    if (!kernel->is_eliminative())
                    {
                        LanguageUnit_p kernel_func_def;
                        if (gnode->get_op_type() == "Result" || gnode->get_op_type() == "Constant")
                        {
                            kernel_func_def = codegenerator::HLSLFunctionFile::convert_from(kernel);
                            kernel_func_def->pwd = projgen->lup_codegen->pwd;
                            kernel_func_def->write_to = projgen->lup_codegen->write_to;
                        }
                        else
                        {
                            kernel_func_def = fu->body_unit;
                        }

                        for (auto& it : fu->dep_unit->local_symbol)
                        {
                            kernel_func_def->require(it.second);
                        }

                        kernel_func_defs[body_str] = make_pair(func_name, kernel_func_def);

                        if (gnode->get_op_type() != "Result" && gnode->get_op_type() != "Constant")
                        {
                            // prepare shader
                            LanguageUnit_p shader_decl = std::make_shared<LanguageUnit>(
                                "declaration::" + hShader_name + "_decl",
                                "void* " + hShader_name + ";\n");
                            string fname = kernel_func_def->symbol;
                            if (fname.length() > 128)
                            {
                                size_t hashcode = std::hash<std::string>{}(fname);
                                fname = "compressed_src_" + std::to_string(hashcode);
                            }

                            std::string file = "file://HLSL/" + fname + m_kernel_suffix;
                            std::string load_str = hShader_name + " = dxShaderLoad(\"" + file +
                                                   "\", &num_inputs, &num_outputs);\nif (!" +
                                                   hShader_name +
                                                   ") {\n    std::cout << \"Invalid Shader Source "
                                                   "for Compilation: " +
                                                   file + "\";\n    exit(1);\n}\n";
                            std::string unload_str = "dxShaderUnload(" + hShader_name + ");\n";
                            LanguageUnit_p shader_load =
                                std::make_shared<LanguageUnit>(hShader_name + "_load", load_str);
                            LanguageUnit_p shader_unload = std::make_shared<LanguageUnit>(
                                hShader_name + "_unload", unload_str);
                            shader_load->require(shader_decl);
                            lup_shaders_load->unit_vec.push_back(shader_load);
                            lup_shaders_unload->unit_vec.push_back(shader_unload);
                        }
                        else
                        {
                            // lup_member->unit_vec.push_back(kernel_func_defs[body_str].second);
                        }
                    }
                }
                else
                {
                    func_name = kernel_func_defs[body_str].first;
                    hShader_name = func_name + "_hShader";
                }
            }

            if (kernel_func_defs.find(body_str) != kernel_func_defs.end())
            {
                lup_func_calls->require(kernel_func_defs[body_str].second);
            }

            // todo: move func call generation to kernel emitter
            NNFUSION_CHECK_NOT_NULLPTR(async_info.execution_stream);
            std::string stream_name = async_info.execution_stream->get_name();
            std::string call_str = fu->call_unit->get_code();
            if (gnode->get_op_type() == "Result" || gnode->get_op_type() == "Constant")
            {
                call_str = func_name + call_str;
                if (kernel && kernel->is_eliminative())
                {
                    call_str = "// " + call_str;
                }
            }
            else
            {
                std::string buffers =
                    "void* args_" + to_string(count) + "[] = { " + call_str + "};\n";
                if (kernel && kernel->is_eliminative())
                {
                    buffers += "// ";
                }
                call_str = buffers + "dxShaderLaunchAsync(" + hShader_name + ", args_" +
                           to_string(count) + ", " + stream_name + ");\n";
            }

            LanguageUnit_p kernel_func_call =
                std::make_shared<LanguageUnit>(fu->call_unit->get_symbol(), call_str);

            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).first);
            auto mem_ref = codegen_mem_ref(kernel);
            if (mem_ref != nullptr)
                lup_func_calls->unit_vec.push_back(mem_ref);
            lup_func_calls->unit_vec.push_back(kernel_func_call);
            if (FLAGS_fcustomized_mem_imp)
                lup_func_calls->unit_vec.push_back(get_customized_mem_imp(ins).second);
            count++;
        }

        if (thread_name != "default_thread")
        {
            nnfusion::errors::NotSupported("HLSL CPP codegen does not support non-default thread.");
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

    separate_func_defs_files(-1, m_kernel_folder);

    return true;
}

void HLSLCPPCodegenPass::generate_main(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_main_content = std::make_shared<LanguageUnit>("main_content");
    lup_main->unit_vec.push_back(lup_main_content);
    auto& lu_ = *lup_main_content;
    lu_ << "\nhlsl_init();\n\n";

    LanguageUnit h2dcopy("h2dcopy");
    LanguageUnit d2hcopy("d2hcopy");
    LanguageUnit fillval("fillval");

    for (size_t i = 0; i < tu->arg.size(); i++)
    {
        auto& tensor = *tu->arg[i];
        //malloc host input arg
        lu_ << "//input argument\n";
        lu_ << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
            << "_host = new " << tensor.get_element_type().c_type_string() << "["
            << tensor.get_tensor_layout()->get_size() << "];\n";
        lu_ << "void* " << tensor.get_name() << " = dxMemAlloc(sizeof("
            << tensor.get_element_type().c_type_string() << ") * "
            << tensor.get_tensor_layout()->get_size() << ");\n";
        fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size() << "; ++i) "
                << tensor.get_name() << "_host[i]= 1;\n";
        h2dcopy << "dxMemcpyHtoDAsync(" << tensor.get_name() << ", " << tensor.get_name()
                << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                << tensor.get_tensor_layout()->get_size() << ", 0);\n";
    }

    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        //malloc host output arg
        lu_ << "//output argument\n";
        lu_ << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
            << "_host = new " << tensor.get_element_type().c_type_string() << "["
            << tensor.get_tensor_layout()->get_size() << "];\n";
        if (FLAGS_fextern_result_memory)
        {
            lu_ << "void* " << tensor.get_name() << " = dxMemAlloc(sizeof("
                << tensor.get_element_type().c_type_string() << ") * "
                << tensor.get_tensor_layout()->get_size() << ");\n";
        }

        d2hcopy << "dxMemcpyDtoHAsync(" << tensor.get_name() << "_host, " << tensor.get_name()
                << ", sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                << tensor.get_tensor_layout()->get_size() << ", nullptr);\n";
    }

    lu_ << "\n// fill input values\n";
    lu_ << fillval.get_code() << "\n";

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

    lu_ << h2dcopy.get_code();
    lu_ << "kernel_entry(" << join(params, ", ") << ");\n";
    lu_ << d2hcopy.get_code();
    lu_ << "dxStreamSynchronize(0);\n";
    lu_ << "std::string result;\n";
    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        // lu_ << "std::cout << \"" << tensor.get_name() << "_host = [\" << " << tensor.get_name()
        //     << "_host[0] << \", \" << " << tensor.get_name() << "_host[1] << \",  .., \" << "
        //     << tensor.get_name() << "_host[" << tensor.get_tensor_layout()->get_size()
        //     << "-1] << \"]\" << std::endl;";
        size_t num = std::min(size_t(10), tensor.get_tensor_layout()->get_size());
        if (num == 1)
        {
            lu_ << "result = \"" << tensor.get_name() << "_host = [\" + std::to_string("
                << tensor.get_name() << "_host[0]) + \"]\\n\";\n";
        }
        else
        {
            lu_ << "result = \"" << tensor.get_name() << "_host = [";
            for (size_t j = 0; j < num; j++)
            {
                lu_ << "\" + std::to_string(" << tensor.get_name() << "_host[" << j << "]) + \", ";
            }
            lu_ << ".., \" + std::to_string(" << tensor.get_name() << "_host["
                << tensor.get_tensor_layout()->get_size() << "-1]) + \"]\\n\";\n";
        }
        lu_ << "OutputDebugStringA(result.c_str());\n";
    }

    lu_ << "\n//free context\n";
    for (size_t i = 0; i < tu->arg.size(); i++)
    {
        auto& tensor = *tu->arg[i];
        lu_ << "dxMemFree(" << tensor.get_name() << ");\n";
    }

    if (FLAGS_fextern_result_memory)
    {
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_ << "dxMemFree(" << tensor.get_name() << ");\n";
        }
    }
    lu_ << "hlsl_free();\n\n";
}

std::string HLSLCPPCodegenPass::get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu)
{
    unordered_set<string> allocated;
    vector<string> params;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        ss << "void* " << tv->get_name();
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }

    for (int i = 0; i < tu->out.size(); i++)
    {
        auto tv = tu->out[i];
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        if (FLAGS_fextern_result_memory)
            ss << "void* " << tv->get_name();
        else
            ss << "void** " << tv->get_name();
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }
    return join(params, ", ");
}

void HLSLCPPCodegenPass::set_global_member(std::shared_ptr<InterpreterContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu)
{
    this->device_async_manager =
        AsyncManagerFactory::get_device_stream_async_manager(tu->m_graph, HLSL);
    this->host_async_manager =
        AsyncManagerFactory::get_host_async_manager(tu->m_graph, GENERIC_CPU);

    // auto& prog = tu->program;
    // for (auto iterator : prog)
    // {
    //     for (auto ins : *iterator)
    //     {
    //         auto kernel = ins->getKernel();
    //         if (!kernel || !kernel->get_or_emit_source())
    //             continue;
    //         for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
    //         {
    //             global_required.insert(it.second->symbol);
    //         }
    //     }
    // }

    return;
}

bool HLSLCPPCodegenPass::after_projgen()
{
    BaseCodegenPass::after_projgen();
    struct stat s;
    std::string cmd;

    std::string main_path = m_codegen_folder + projgen->lup_codegen->write_to;
    std::string Direct3DWinNN_path = m_codegen_folder + std::string("Direct3DWinNN/");
    std::string Direct3DXBoxNN_path = m_codegen_folder + std::string("Direct3DXBoxNN/");
    if (stat(main_path.c_str(), &s) == 0)
    {
        cmd = std::string("cp -f ") + main_path + " " + Direct3DWinNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy Main.cpp to Direct3DWinNN folder.\n");
        }

        cmd = std::string("cp -f ") + main_path + " " + Direct3DXBoxNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy Main.cpp to Direct3DXBoxNN folder.\n");
        }

        cmd = std::string("rm -f ") + main_path;
        if (0 != system(cmd.c_str()))
        {
            NNFUSION_LOG(INFO) << get_current_dir_name() << main_path;
            throw nnfusion::errors::RuntimeError("Failed to remove Main.cpp.\n");
        }
    }
    else
    {
        throw nnfusion::errors::RuntimeError("Failed to codegen Main.cpp.\n");
    }

    std::string constant_folder = m_codegen_folder + std::string("Constant/");
    if (stat(constant_folder.c_str(), &s) == 0)
    {
        cmd = std::string("cp -rf ") + constant_folder + " " + Direct3DWinNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy Constant folder to Direct3DWinNN folder.\n");
        }

        cmd = std::string("cp -rf ") + constant_folder + " " + Direct3DXBoxNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy Constant folder to Direct3DXBoxNN folder.\n");
        }

        cmd = std::string("rm -rf ") + constant_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to remove Constant folder.\n");
        }
    }

    if (stat(m_kernel_folder.c_str(), &s) == 0)
    {
        cmd = std::string("cp -rf ") + m_kernel_folder + " " + Direct3DWinNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy kernel folder to Direct3DWinNN folder.\n");
        }

        cmd = std::string("cp -rf ") + m_kernel_folder + " " + Direct3DXBoxNN_path;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError(
                "Failed to copy kernel folder to Direct3DXBoxNN folder.\n");
        }

        cmd = std::string("rm -rf ") + m_kernel_folder;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to remove kernel folder.\n");
        }
    }

    std::string vs_folder1 = Direct3DWinNN_path + std::string(".vs/");
    if (stat(vs_folder1.c_str(), &s) == 0)
    {
        cmd = std::string("rm -rf ") + vs_folder1;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to remove .vs folder.\n");
        }
    }

    std::string vs_folder2 = Direct3DXBoxNN_path + std::string(".vs/");
    if (stat(vs_folder2.c_str(), &s) == 0)
    {
        cmd = std::string("rm -rf ") + vs_folder2;
        if (0 != system(cmd.c_str()))
        {
            throw nnfusion::errors::RuntimeError("Failed to remove .vs folder.\n");
        }
    }
    return true;
}