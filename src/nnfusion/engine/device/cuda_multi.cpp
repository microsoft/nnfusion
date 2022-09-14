// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_multi.hpp"
#include "nnfusion/engine/pass/codegen/cuda_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/cuda_multi_codegen_pass.hpp"
#include "nnfusion/frontend/util/parameter.hpp"

using namespace nnfusion;
using namespace nnfusion::engine;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass;

DECLARE_string(params);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fhost_entry);

CudaMultiEngine::CudaMultiEngine()
    : CudaEngine()
{
    this->erase_all_codegen();
}

static void remove_extern_c(std::string f)
{
    std::ifstream file(f, std::ios::out);
    if (!file.is_open())
        return;
    std::string line;
    std::stringstream myself;
    std::string extern_c = "extern \"C\"";
    while (std::getline(file, line))
    {
        while (line.find(extern_c) < line.length())
            line.erase(line.find(extern_c), extern_c.size());
        myself << line << "\n";
    }
    file.close();

    std::ofstream ofile(f);
    ofile << myself.str();
    ofile.close();
}

static std::string get_kernel_entry_paras(std::shared_ptr<TranslationUnit> tu, bool is_host)
{
    unordered_set<string> allocated;
    vector<string> params;
    for (int i = 0; i < tu->arg.size(); i++)
    {
        auto tv = tu->arg[i];
        string type = tv->get_element_type().c_type_string();
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
        string type = tv->get_element_type().c_type_string();
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

static std::string get_kernel_entry_args(std::shared_ptr<TranslationUnit> tu, bool is_host)
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
        args.push_back(name);
    }
    return join(args, ", ");
}

bool CudaMultiEngine::run_on_graphs(std::vector<graph::Graph::Pointer> graphs,
                                    EngineContext::Pointer pass_context)
{
    std::vector<CodeGenerator::Pointer> proj_gens;
    std::unordered_map<std::string, size_t> pool_size;
    std::string arg_list, arg_vars, codegen_folder, write_to, header_write_to;
    int graph_cnt = 0;
    for (auto& graph : graphs)
    {
        // Wrapp code in "namespace graph_0{"
        // "} //namespace graph_0"
        std::string graph_name = "graph_" + to_string(graph_cnt);
        auto context = make_shared<EngineContext>();
        this->run_on_graph(graph, context);
        CudaMultiCodegenPassPre cmcpp;
        cmcpp.run(context->m_legacy_ctx, context->m_legacy_tu);
        proj_gens.push_back(cmcpp.get_projgen());
        proj_gens.back()->codegen_with_preprocess(graph_cnt == 0, graph_name);

        auto& allocator_list = context->m_legacy_tu->memory_allocator_factory->get_allocator_list();
        for (auto alloc : allocator_list)
        {
            if (pool_size.find(alloc.second->get_name()) != pool_size.end())
                pool_size[alloc.second->get_name()] = 0;
            pool_size[alloc.second->get_name()] =
                max(alloc.second->max_allocated(), pool_size[alloc.second->get_name()]);
        }

        if (graph_cnt == 0)
        {
            arg_list = get_kernel_entry_paras(context->m_legacy_tu, false);
            arg_vars = get_kernel_entry_args(context->m_legacy_tu, false);
            codegen_folder = proj_gens.back()->get_codegen_folder();
            write_to = proj_gens.back()->lup_exec->write_to;
            header_write_to =
                proj_gens.back()->lup_codegen->local_symbol["codegen_header"]->write_to;
        }

        if (graph_cnt + 1 == graphs.size())
        {
            remove_extern_c(codegen_folder + write_to);

            {
                LanguageUnit global_init;
                LanguageUnit global_free;
                LanguageUnit global_device_type;
                LanguageUnit global_workspace_size;
                global_init << "extern \"C\" void cuda_init()\n{\n";
                graph_cnt = 0;
                size_t workspace_size = 0;
                for (auto graph_cnt = 0; graph_cnt < graphs.size(); graph_cnt++)
                {
                    std::string graph_name = "graph_" + to_string(graph_cnt);
                    for (auto pool : pool_size)
                    {
                        if (graph_cnt == 0)
                        {
                            global_init << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << graph_name
                                        << "::" << pool.first << "_memory_pool," << pool.second
                                        << "));\n";
                            workspace_size += pool.second;
                        }
                        else
                        {
                            global_init << graph_name << "::" << pool.first << "_memory_pool = "
                                        << "graph_0::" << pool.first << "_memory_pool;"
                                        << "\n";
                        }
                    }
                    global_init << graph_name << "::cuda_init();\n";
                }
                global_init << "}\n";
                global_free << "extern \"C\" void cuda_free() {graph_0::cuda_free();}\n";
                global_device_type << "int get_device_type() { return 0; }\n";
                global_workspace_size << "size_t get_workspace_size() { return " << workspace_size
                                      << "; }\n";

                global_init.write_to = global_free.write_to = global_device_type.write_to =
                    global_workspace_size.write_to = write_to;
                global_init.pwd = global_free.pwd = global_device_type.pwd =
                    global_workspace_size.pwd = codegen_folder;
                global_init.execute();
                global_free.execute();
                global_device_type.execute();
                global_workspace_size.execute();
            }

            {
                LanguageUnit global_sym_methods;
                LanguageUnit global_sym_defs;
                LanguageUnit global_entry;
                global_entry.pwd = codegen_folder;
                global_entry.write_to = write_to;
                auto vec_dim_params =
                    nnfusion::frontend::build_multi_onnx_params_from_string(FLAGS_params);
                global_entry << "extern \"C\" int kernel_entry(";
                global_entry << arg_list;
                global_entry << ")\n{\n";
                graph_cnt = 0;
                auto first_params = vec_dim_params[0];
                for (auto param : first_params)
                {
                    global_sym_defs << "extern \"C\" void set_" << param.first << "(int);\n"
                                    << "extern \"C\" int get_" << param.first << "();\n";
                    global_sym_methods << "int " << param.first << ";\n"
                                       << "extern \"C\" void set_" << param.first << "(int s) { "
                                       << param.first << " = s; }\n"
                                       << "extern \"C\" int get_" << param.first << "() { return "
                                       << param.first << "; }\n";
                }
                for (auto dim_params : vec_dim_params)
                {
                    std::string condition = "";
                    for (auto param : dim_params)
                    {
                        if (!condition.empty())
                            condition += " && ";
                        condition += "get_" + param.first + "() == " + param.second.sym();
                    }
                    global_entry << "if(" << condition << ")\n{\n";
                    global_entry << "graph_" << graph_cnt << "::kernel_entry(" + arg_vars + ");\n";
                    global_entry << "}\n";
                    graph_cnt++;
                }
                global_entry << "return 0;\n}\n";
                global_entry.execute();

                global_entry << "\n" << global_sym_methods.get_code();
                global_entry.execute();

                global_sym_defs.write_to = header_write_to;
                global_sym_defs.pwd = codegen_folder;
                global_sym_defs.execute();
            }
            cmcpp.invoke_after_projgen();
            break;
        }

        graph_cnt++;
    }

    return true;
}

bool CudaMultiEngine::erase_all_codegen()
{
    bool has_codegen = true;
    while (has_codegen)
    {
        has_codegen = false;
        for (auto i = m_passes->begin(); i != m_passes->end(); i++)
        {
            auto p = *i;
            if (dynamic_pointer_cast<BaseCodegenPass>(p) != nullptr)
            {
                m_passes->erase(i);
                has_codegen = true;
                break;
            }
        }
    }
    return true;
}