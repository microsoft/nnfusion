// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_cs_codegen_pass.hpp"
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
DECLARE_bool(fhost_entry);
DECLARE_bool(ffunction_codegen);

void HLSLCSCodegenPass::initialize(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    set_global_member(ctx, tu);
    // setup lup_codegen execution info
    projgen->lup_codegen->pwd = m_codegen_folder;
    projgen->lup_codegen->write_to = "nnfusion_rt.cs";
    auto& copy_templates = projgen->lup_codegen->copy_templates;
    copy_templates.emplace_back("dxcompute/make.bat", "./make.bat");

    projgen->lup_codegen->require(lup_program);
    projgen->lup_codegen->remove(projgen->lup_init);
    projgen->lup_codegen->remove(projgen->lup_exec);
    projgen->lup_codegen->remove(projgen->lup_exit);
    projgen->lup_exit->remove(projgen->lup_exec);
    projgen->lup_exec->remove(projgen->lup_init);
    lup_program->unit_vec.push_back(lup_member);
    lup_program->unit_vec.push_back(projgen->lup_init);
    lup_program->unit_vec.push_back(projgen->lup_exec);
    lup_program->unit_vec.push_back(projgen->lup_exit);
    lup_program->unit_vec.push_back(lup_main);

    // setup main_block
    auto& lu_init_begin = *(projgen->lup_init->begin);
    {
        lu_init_begin << "\nstatic void hlsl_init()\n{\n";
    }

    auto& lu_init_end = *(projgen->lup_init->end);
    {
        lu_init_end << get_sync()->get_code();
        lu_init_end << "}\n\n";
    }

    auto& lu_exec_begin = *(projgen->lup_exec->begin);
    {
        std::string params = get_kernel_entry_paras(tu);
        lu_exec_begin << "\nstatic int kernel_entry(" << params << ")\n{\n";
    }

    auto& lu_exec_end = *(projgen->lup_exec->end);
    {
        lu_exec_end << "return 0;\n";
        lu_exec_end << "}\n\n";
    }

    if (FLAGS_fhost_entry)
    {
        fill_exec_host(tu);
    }
    auto& lu_exit_begin = *(projgen->lup_exit->begin);
    {
        lu_exit_begin << "\nstatic void hlsl_free()\n{\n";
    }

    auto& lu_exit_end = *(projgen->lup_exit->end);
    {
        lu_exit_end << "UnloadHlslImportedDll();\n}\n\n";
    }

    auto& lu_main_begin = *(lup_main->begin);
    {
        lu_main_begin << "\nstatic int Main(string[] args)\n{\n";
    }

    auto& lup_main_end = *(lup_main->end);
    {
        lup_main_end << "return 0;\n";
        lup_main_end << "}\n\n";
    }

    auto& lup_program_begin = *(lup_program->begin);
    {
        lup_program_begin << "\nclass Program\n{\n";
    }

    auto& lup_program_end = *(lup_program->end);
    {
        lup_program_end << "}\n\n";
    }

    lup_member->unit_vec.push_back(declaration::antares_hlsl_dll_cs);

    // add requirement
    projgen->lup_codegen->require(header::systems);

    generate_main(ctx, tu);

    return;
}

bool HLSLCSCodegenPass::collect_funcs(std::shared_ptr<InterpreterContext> ctx,
                                      std::shared_ptr<TranslationUnit> tu)
{
    if (!tu)
        return false;

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
                                "static IntPtr " + hShader_name + ";\n");
                            string fname = kernel_func_def->symbol;
                            if (fname.length() > 128)
                            {
                                size_t hashcode = std::hash<std::string>{}(fname);
                                fname = "compressed_src_" + std::to_string(hashcode);
                            }

                            std::string file = "file://HLSL/" + fname + m_kernel_suffix;
                            std::string load_str = hShader_name + " = dxShaderLoad(\"" + file +
                                                   "\");\nif (" + hShader_name +
                                                   " == IntPtr.Zero)\n    throw new  "
                                                   "Exception(\"Invalid Shader Source "
                                                   "for Compilation: " +
                                                   file + "\");\n";
                            LanguageUnit_p shader_load =
                                std::make_shared<LanguageUnit>(hShader_name + "_load", load_str);
                            lup_member->unit_vec.push_back(shader_decl);
                            projgen->lup_init->unit_vec.push_back(shader_load);
                        }
                        else
                        {
                            lup_member->unit_vec.push_back(kernel_func_defs[body_str].second);
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
                if (gnode->get_op_type() == "Result" || gnode->get_op_type() == "Constant")
                {
                    // do not add requre
                }
                else
                {
                    lup_func_calls->require(kernel_func_defs[body_str].second);
                }
            }

            // todo: move func call generation to kernel emitter
            NNFUSION_CHECK_NOT_NULLPTR(async_info.execution_stream);
            std::string stream_name = async_info.execution_stream->get_name();
            std::string call_str = fu->call_unit->get_code();
            if (gnode->get_op_type() == "Result" || gnode->get_op_type() == "Constant")
            {
                call_str = func_name + call_str;
            }
            else
            {
                call_str = "dxShaderLaunchAsync(" + hShader_name + ", new IntPtr[]{" + call_str +
                           "}, " + stream_name + ");\n";
            }

            if (kernel && kernel->is_eliminative())
            {
                call_str = "// " + call_str;
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
        }

        if (thread_name != "default_thread")
        {
            nnfusion::errors::NotSupported("HLSL CS codegen does not support non-default thread.");
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

bool HLSLCSCodegenPass::collect_mem(std::shared_ptr<InterpreterContext> ctx,
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
    for (const auto& allocator : allocator_list)
    {
        total_alloc += allocator.second->max_allocated();
    }
    LanguageUnit_p total = std::make_shared<LanguageUnit>(
        "total_memory", "// total memory:" + to_string(total_alloc) + "\n");
    lup_mem_alloc->unit_vec.push_back(total);

    size_t offset = 0;
    for (const auto& allocator : allocator_list)
    {
        auto init = allocator.second->emit_memory_init();
        auto alloc = allocator.second->emit_memory_alloc();
        auto free = allocator.second->emit_memory_free();

        lup_member->unit_vec.push_back(init);
        if (FLAGS_ffunction_codegen)
        {
            auto mempool_offset = allocator.second->emit_memory_pool_offset(offset);
            offset += allocator.second->max_allocated();
            lup_mem_alloc->unit_vec.push_back(mempool_offset);
        }
        lup_mem_alloc->unit_vec.push_back(alloc);
        // lup_mem_alloc->require(init);
        lup_mem_free->unit_vec.push_back(free);
        // lup_mem_free->require(init);
    }

    return true;
}

void HLSLCSCodegenPass::generate_main(std::shared_ptr<InterpreterContext> ctx,
                                      std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p lup_main_content = std::make_shared<LanguageUnit>("main_content");
    lup_main->unit_vec.push_back(lup_main_content);
    auto& lu_ = *lup_main_content;
    lu_ << "\nhlsl_init();\n\n";

    LanguageUnit fillval("fillval");

    for (size_t i = 0; i < tu->arg.size(); i++)
    {
        auto& tensor = *tu->arg[i];
        //malloc host input arg
        lu_ << "//input argument\n";
        lu_ << "var " << tensor.get_name() << "_host = new "
            << tensor.get_element_type().c_type_string() << "["
            << tensor.get_tensor_layout()->get_size() << "];\n";
        lu_ << "var " << tensor.get_name() << " = dxMemAlloc(sizeof("
            << tensor.get_element_type().c_type_string() << ") * "
            << tensor.get_tensor_layout()->get_size() << ");\n";
        fillval << "for (int i = 0; i < " << tensor.get_name() << "_host.Length; ++i) "
                << tensor.get_name() << "_host[i]= 1;\n";
    }

    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        //malloc host output arg
        lu_ << "//output argument\n";
        lu_ << "var " << tensor.get_name() << "_host = new "
            << tensor.get_element_type().c_type_string() << "["
            << tensor.get_tensor_layout()->get_size() << "];\n";
    }

    lu_ << "\n// fill input values\n";
    lu_ << fillval.get_code() << "\n";

    std::string args = get_kernel_entry_args(tu);

    lu_ << get_h2dcopy(tu)->get_code();
    lu_ << "kernel_entry(" << args << ");\n";
    lu_ << get_d2hcopy(tu)->get_code();
    lu_ << get_sync()->get_code();

    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        lu_ << "Console.WriteLine(\"" << tensor.get_name() << "_host = [\" + " << tensor.get_name()
            << "_host[0] + \", \" + " << tensor.get_name() << "_host[1] + \",  .., \" + "
            << tensor.get_name() << "_host[" << tensor.get_name() << "_host.Length-1]+ \"]\");";
    }

    lu_ << "\n//free context\n";
    lu_ << "hlsl_free();\n\n";
}

std::string HLSLCSCodegenPass::get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu,
                                                      bool is_host)
{
    unordered_set<string> allocated;
    vector<string> params;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        // ss << type << "* " << tv->get_name();
        ss << "IntPtr " << tv->get_name();
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
        string type = tv->get_element_type().c_type_string();
        stringstream ss;
        // ss << type << "* " << tv->get_name();
        ss << "IntPtr " << tv->get_name();
        if (is_host)
        {
            ss << "_host";
        }
        allocated.insert(tv->get_name());
        params.push_back(ss.str());
    }
    return join(params, ", ");
}

void HLSLCSCodegenPass::set_global_member(std::shared_ptr<InterpreterContext> ctx,
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

LanguageUnit_p HLSLCSCodegenPass::get_d2hcopy(std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p d2hcopy = std::make_shared<LanguageUnit>("d2hcopy");
    for (size_t i = 0; i < tu->out.size(); i++)
    {
        auto& tensor = *tu->out[i];
        *d2hcopy << "dxMemcpyDtoHAsync(Marshal.UnsafeAddrOfPinnedArrayElement(" << tensor.get_name()
                 << "_host, 0), " << tensor.get_name() << ", sizeof("
                 << tensor.get_element_type().c_type_string() << ") * "
                 << tensor.get_tensor_layout()->get_size() << ", IntPtr.Zero);\n";
    }
    return d2hcopy;
}

LanguageUnit_p HLSLCSCodegenPass::get_h2dcopy(std::shared_ptr<TranslationUnit> tu)
{
    LanguageUnit_p h2dcopy = std::make_shared<LanguageUnit>("h2dcopy");

    for (size_t i = 0; i < tu->arg.size(); i++)
    {
        auto& tensor = *tu->arg[i];
        *h2dcopy << "dxMemcpyHtoDAsync(" << tensor.get_name()
                 << ", Marshal.UnsafeAddrOfPinnedArrayElement(" << tensor.get_name()
                 << "_host, 0), sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                 << tensor.get_tensor_layout()->get_size() << ", IntPtr.Zero);\n";
    }
    return h2dcopy;
}

LanguageUnit_p HLSLCSCodegenPass::get_sync()
{
    return std::make_shared<LanguageUnit>("device_sync", "dxStreamSynchronize(IntPtr.Zero);\n");
}