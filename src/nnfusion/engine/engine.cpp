// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "engine.hpp"

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