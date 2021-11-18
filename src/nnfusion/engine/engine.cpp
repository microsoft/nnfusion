// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "engine.hpp"
#include "nnfusion/engine/pass/codegen/base_codegen_pass.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"

DEFINE_bool(fkernels_as_files, false, "Saving kernels as standalone source code files.");
DEFINE_int64(fkernels_files_number, -1, "Saving kernels into how many source code files.");
DEFINE_bool(ftraining_mode, false, "Turn on training mode.");
DEFINE_bool(fextern_result_memory, false, "Model result tensor memory is managed externally.");
DEFINE_int32(fwarmup_step, 5, "Warm up step.");
DEFINE_int32(frun_step, 100, "Run step.");
DEFINE_int32(min_log_level,
             1,
             "Minimum logging level: 0 - debug; 1 - info; 2 - warning; 3 - error; 4 - fatal;");
DEFINE_bool(fcustomized_mem_imp, false, "Use customized memory implementation in codegen files;");
DEFINE_bool(fhost_entry, false, "provide entry on host memory");
DEFINE_bool(fuse_cpuprofiler, false, "");
using namespace nnfusion;

Engine::Engine()
{
    m_passes = make_shared<InterpreterPassManager>();
    g_passes = make_shared<GraphPassManager>();
    g_visitor = make_shared<GraphVisitor>();
}

bool Engine::run_on_graph(graph::Graph::Pointer graph, EngineContext::Pointer context)
{
    if (context == nullptr)
        context = make_shared<EngineContext>();

    NNFUSION_LOG(INFO) << "Graph Passes count:" << (g_passes != nullptr ? g_passes->size() : 0);
    NNFUSION_LOG(INFO) << "Interpreter Passes count:"
                       << (m_passes != nullptr ? m_passes->size() : 0);

    bool result = true;
    if (g_passes != nullptr)
        result = g_passes->run_on_graph(graph, context);

    NNFUSION_CHECK(result) << "Engine failed after finished graph passes.";

    ir::Program::Pointer p = nullptr;
    if (g_visitor != nullptr)
        p = g_visitor->run_on_graph(graph, context);
    else
        return result;

    NNFUSION_CHECK(p != nullptr) << "Engine failed after finished graph visitor.";

    if (m_passes != nullptr)
        result = m_passes->run_on_program(p, context);

    NNFUSION_CHECK(result) << "Engine failed after finished codegen passes.";
    return result;
}

nnfusion::TranslationUnit::Pointer Engine::convert_graph_to_program(graph::Graph::Pointer graph)
{
    // this function has the same logic with Engine::run_on_graph, 
    // except that it will not do codegen and will return translation unit object.
    auto context = make_shared<EngineContext>();
    bool result = true;
    if (g_passes != nullptr)
        result = g_passes->run_on_graph(graph, context);
    NNFUSION_CHECK(result) << "Engine failed after finished graph passes.";
    ir::Program::Pointer p = nullptr;
    p = g_visitor->run_on_graph(graph, context);

    shared_ptr<TranslationUnit> tu(new TranslationUnit());
    shared_ptr<InterpreterContext> ctx(new InterpreterContext());
    tu->program = move(*p);
    {
        NNFUSION_LOG(INFO) << "Legacy graph used in interpreter;";
        tu->m_graph = context->m_legacy_graph;
        NNFUSION_CHECK(tu->m_graph != nullptr);
        std::unordered_set<graph::Graph::Pointer> graph_vec{tu->m_graph};
        ctx->m_graphs = graph_vec;
        tu->blacklist = context->blacklist;
    }
    // neglect the codegen pass
    for (size_t i = 0; i < m_passes->size() - 1; i++)
    {
        result = (*m_passes)[i]->run(ctx, tu);
        if (!result)
            break;
    }
    return tu;
}
