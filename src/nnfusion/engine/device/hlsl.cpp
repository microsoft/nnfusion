// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl.hpp"
#include "degree_based_visitor.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_cpp_codegen_pass.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_cs_codegen_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_async_info_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_layout_pass.hpp"
#include "nnfusion/engine/pass/graph/autodiff_pass.hpp"
#include "nnfusion/engine/pass/graph/batchnorm_inference_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/blockfusion_pass.hpp"
#include "nnfusion/engine/pass/graph/common_subexpression_elimination_pass.hpp"
#include "nnfusion/frontend/util/parameter.hpp"
#include "reversed_dfs_visitor.hpp"

#include "nnfusion/engine/pass/graph/gemm_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/ir_based_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_profiling_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/kernel_tuning.hpp"
#include "nnfusion/engine/pass/graph/multi_reshape_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/pass/graph/pattern_substitution.hpp"
#include "nnfusion/engine/pass/graph/reduce_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/vector_dot_transpose_pass.hpp"

#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/tensor/inplace_tensor_analysis.hpp"
#include "nnfusion/engine/pass/tensor/liveness_analysis.hpp"
#include "nnfusion/engine/pass/tensor/tensor_device_dispatcher.hpp"
#include "nnfusion/engine/pass/tensor/tensor_memory_layout.hpp"

using namespace nnfusion;
using namespace nnfusion::engine;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass;

DEFINE_string(fhlsl_codegen_type,
              "default",
              "choose hlsl codegen type from [default(will be deprecated), csharp, cpp]");
HLSLEngine::HLSLEngine()
    : Engine()
{
    if (FLAGS_fhlsl_codegen_type == "csharp")
    {
        g_passes->push_back(make_shared<CSEPass>());
        g_passes->push_back(make_shared<AutodiffPass>());
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
        g_passes->push_back(make_shared<VectorDotTransposePass>());
        g_passes->push_back(make_shared<GemmFusionPass>());
        g_passes->push_back(make_shared<BatchNormInferenceFoldingPass>());
        g_passes->push_back(make_shared<AssignLayoutPass>());
        g_passes->push_back(make_shared<OpInplacePass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Assign stream passes
        g_passes->push_back(make_shared<AssignAsyncInfoPass>());

        // Visitor
        // g_visitor = make_shared<DegreeBasedVisitor>();
        g_visitor = make_shared<ReversedDFSVisitor>();

        // extract graph signature
        m_passes->push_back(make_shared<ExtractGraphSignature>());
        // Do tensor allocation plan
        m_passes->push_back(make_shared<TensorDeviceDispatcher>());
        m_passes->push_back(make_shared<TensorLivenessAnalysis>());
        m_passes->push_back(make_shared<InplaceTensorAnalysis>());
        m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

        // Do codegen
        m_passes->push_back(make_shared<HLSLCSCodegenPass>());
    }
    else if (FLAGS_fhlsl_codegen_type == "cpp")
    {
        g_passes->push_back(make_shared<CSEPass>());
        g_passes->push_back(make_shared<AutodiffPass>());
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
        g_passes->push_back(make_shared<VectorDotTransposePass>());
        g_passes->push_back(make_shared<GemmFusionPass>());
        g_passes->push_back(make_shared<BatchNormInferenceFoldingPass>());
        g_passes->push_back(make_shared<AssignLayoutPass>());
        g_passes->push_back(make_shared<OpInplacePass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Assign stream passes
        g_passes->push_back(make_shared<AssignAsyncInfoPass>());

        // Visitor
        // g_visitor = make_shared<DegreeBasedVisitor>();
        g_visitor = make_shared<ReversedDFSVisitor>();

        // extract graph signature
        m_passes->push_back(make_shared<ExtractGraphSignature>());
        // Do tensor allocation plan
        m_passes->push_back(make_shared<TensorDeviceDispatcher>());
        m_passes->push_back(make_shared<TensorLivenessAnalysis>());
        m_passes->push_back(make_shared<InplaceTensorAnalysis>());
        m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

        // Do codegen
        m_passes->push_back(make_shared<HLSLCPPCodegenPass>());
    }
    else
    {
        g_passes->push_back(make_shared<GradientWeightMappingPass>());
        g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
        g_passes->push_back(make_shared<ReduceFusionPass>());
        g_passes->push_back(make_shared<IRBasedFusionPass>());

        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<KernelFusionPass>());
        g_passes->push_back(make_shared<KernelTuning>());
        g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
        g_passes->push_back(make_shared<FetchBasedSelector>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        // Visitor
        g_visitor = make_shared<DegreeBasedVisitor>();

        // Do codegen
        m_passes->push_back(make_shared<HLSLCodegenPass>());
    }
}

bool HLSLMultiEngine::erase_all_codegen()
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

DECLARE_string(params);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fhost_entry);

HLSLMultiEngine::HLSLMultiEngine()
    : HLSLEngine()
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
        ss << "void* " << tv->get_name();
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
            ss << "void* " << tv->get_name();
        else
            ss << "void** " << tv->get_name();
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

bool HLSLMultiEngine::run_on_graphs(std::vector<graph::Graph::Pointer> graphs,
                                    EngineContext::Pointer pass_context)
{
    std::vector<CodeGenerator::Pointer> proj_gens;
    std::vector<std::unordered_map<std::string, size_t>> vec_pool_size(graphs.size());
    std::string arg_list, arg_vars, codegen_folder, write_to, header_write_to;
    int graph_cnt = 0;
    for (auto& graph : graphs)
    {
        // Wrapp code in "namespace graph_0{"
        // "} //namespace graph_0"
        std::string graph_name = "graph_" + to_string(graph_cnt);
        auto context = make_shared<EngineContext>();
        this->run_on_graph(graph, context);
        std::string the_codegen_folder = "./nnfusion_rt/dxcompute_codegen/";
        std::string kernel_folder =
            "./nnfusion_rt/dxcompute_codegen/HLSL_" + to_string(graph_cnt) + "/";
        std::string kernel_suffix = ".hlsl";
        HLSLMultiCodegenPassPre cmcpp(the_codegen_folder, kernel_folder, kernel_suffix);
        cmcpp.run(context->m_legacy_ctx, context->m_legacy_tu);
        proj_gens.push_back(cmcpp.get_projgen());
        proj_gens.back()->codegen_with_preprocess(graph_cnt == 0, graph_name);

        auto& allocator_list = context->m_legacy_tu->memory_allocator_factory->get_allocator_list();
        for (auto alloc : allocator_list)
        {
            vec_pool_size[graph_cnt][alloc.second->get_name()] = alloc.second->max_allocated();
        }

        if (graph_cnt == 0)
        {
            arg_list = get_kernel_entry_paras(context->m_legacy_tu, false);
            arg_vars = get_kernel_entry_args(context->m_legacy_tu, false);
            codegen_folder = proj_gens.back()->get_codegen_folder();
            write_to = proj_gens.back()->lup_codegen->write_to;
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
                global_init << "extern \"C\" void hlsl_init()\n";
                global_free << "extern \"C\" void hlsl_free() { \n";
                size_t workspace_size = 0;
                global_init.block_begin();
                for (graph_cnt = 0; graph_cnt < graphs.size(); graph_cnt++)
                {
                    std::string graph_name = "graph_" + to_string(graph_cnt);
                    for (auto pool : vec_pool_size[graph_cnt])
                    {
                        if (graph_cnt == 0)
                        {
                            global_init << graph_name << "::" << pool.first
                                        << "_memory_pool = dxMemAlloc(" << pool.second << ");\n";
                            workspace_size += pool.second;
                        }
                        else
                        {
                            if(pool.first.find("persist") < pool.first.length())
                            {
                                global_init << graph_name << "::" << pool.first << "_memory_pool = "
                                        << "graph_0::" << pool.first << "_memory_pool;"
                                        << "\n";
                            }
                            else 
                            {
                                global_init << graph_name << "::" << pool.first
                                        << "_memory_pool = dxMemAlloc(" << pool.second << ");\n";
                                workspace_size += pool.second;
                                global_free << "\tdxMemFree(" << graph_name << "::" << pool.first << "_memory_pool);\n";
                            }
                        }
                    }
                    global_init << graph_name << "::hlsl_init();\n";
                }
                global_init.block_end();
                global_init << "\n";
                global_free << "\tgraph_0::hlsl_free();\n}\n\n";
                global_device_type << "int get_device_type() { return 3; }\n\n";
                global_workspace_size << "size_t get_workspace_size() { return " << workspace_size
                                      << "; }\n\n";

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
                global_entry << ")\n";
                global_entry.block_begin();
                graph_cnt = 0;
                auto first_params = vec_dim_params[0];
                for (auto param : first_params)
                {
                    global_sym_defs << "extern \"C\" RUNTIME_API void set_" << param.first << "(int64_t);\n"
                                    << "extern \"C\" RUNTIME_API int64_t get_" << param.first << "();\n";
                    global_sym_methods << "int64_t " << param.first << ";\n"
                                       << "extern \"C\" void set_" << param.first << "(int64_t s) { "
                                       << param.first << " = s; }\n"
                                       << "extern \"C\" int64_t get_" << param.first << "() { return "
                                       << param.first << "; }\n";
                }
                for (auto dim_params : vec_dim_params)
                {
                    std::string condition = "";
                    for (auto param : dim_params)
                    {
                        if (!condition.empty())
                            condition += " && ";
                        if(param.second.min() == 0)
                            condition += "get_" + param.first + "() == " + to_string(param.second.max());
                        else
                            condition += "get_" + param.first + "() >=" + to_string(param.second.min()) 
                                + " && " + " get_" + param.first + "() <=" + to_string(param.second.max());
                    }
                    global_entry << "if(" << condition << ")\n{\n";
                    global_entry << "\tgraph_" << graph_cnt
                                 << "::kernel_entry(" + arg_vars + ");\n";
                    global_entry << "}\n";
                    graph_cnt++;
                }
                for(auto tv: context->m_legacy_tu->out)
                {
                    if(tv->get_shape().is_dynamic())
                    {
                        auto& dynshape = tv->get_shape().sym_shape;
                        int dim = 0;
                        for(auto sym : *dynshape)
                        {
                            if(sym.is_dynamic())
                            {
                                auto dim_name = tv->get_name() + "_dim_" + to_string(dim);
                                global_sym_defs << "extern \"C\" RUNTIME_API int64_t get_" << dim_name << "();\n";
                                global_sym_methods
                                       << "extern \"C\" int64_t get_" << dim_name << "() { return "
                                       << sym.sym() << "; }\n";
                            }
                        }
                    }
                }
                global_entry << "return 0;\n";
                global_entry.block_end();
                global_entry << "\n" << global_sym_methods.get_code();
                global_entry.execute();

                global_sym_defs.write_to = header_write_to;
                global_sym_defs.pwd = codegen_folder;
                global_sym_defs.execute();
            }

            cmcpp.invoke_after_projgen();
            for (int c = 0; c < graph_cnt; c++)
            {
                std::string kernel_folder =
                    "./nnfusion_rt/dxcompute_codegen/HLSL_" + to_string(c) + "/";
                cmcpp.move_kernel_folder(kernel_folder);
            }
            break;
        }

        graph_cnt++;
    }

    return true;
}