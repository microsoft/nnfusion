// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/core/IR/program.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "nnfusion/engine/pass/graph/graph_pass_base.hpp"

/*
Basically, this engine has three parts:
    1. Run graph passes on graph;
    2. Translate Graph into Program;
    3. Run interpreter passes on Program;

By default, inside Program the execution order is determined, thus it's simple
to do tensor analysis and do memory plan.
*/

namespace nnfusion
{
    class EngineContext : public ir::Tagable
    {
    public:
        using Pointer = shared_ptr<EngineContext>;
        graph::Graph::Pointer m_legacy_graph;
        std::unordered_set<std::shared_ptr<graph::GNode>> blacklist;
    };

    class GraphVisitor
    {
    public:
        using Pointer = shared_ptr<GraphVisitor>;
        virtual nnfusion::ir::Program::Pointer
            run_on_graph(shared_ptr<graph::Graph> graph, EngineContext::Pointer context = nullptr)
        {
            return nullptr;
        }
    };

    class InterpreterPassManager : public vector<shared_ptr<IInterpreterPass>>
    {
    public:
        using Pointer = shared_ptr<InterpreterPassManager>;
        virtual bool run_on_program(ir::Program::Pointer prog,
                                    EngineContext::Pointer context = nullptr)
        {
            // For compatible purpose
            shared_ptr<TranslationUnit> _tu(new TranslationUnit());
            shared_ptr<InterpreterContext> ctx(new InterpreterContext());
            _tu->program = move(*prog);
            bool status = true;

            //\todo(wenxh) Lagacy code - to be removed soon;
            if (context != nullptr)
            {
                NNFUSION_LOG(INFO) << "Legacy graph used in interpreter;";
                _tu->m_graph = context->m_legacy_graph;
                NNFUSION_CHECK(_tu->m_graph != nullptr);
                std::unordered_set<graph::Graph::Pointer> graph_vec{_tu->m_graph};
                ctx->m_graphs = graph_vec;
                _tu->blacklist = context->blacklist;
            }

            for (auto& pass : *this)
            {
                // \todo(wenxh): make this interface fit (prog, context);
                status = pass->run(ctx, _tu);
                if (!status)
                    break;
            }
            return status;
        }
    };

    class GraphPassManager : public vector<shared_ptr<nnfusion::pass::graph::GraphPassBase>>
    {
    public:
        using Pointer = shared_ptr<GraphPassManager>;
        virtual bool run_on_graph(graph::Graph::Pointer graph,
                                  EngineContext::Pointer context = nullptr)
        {
            bool status;
            if (context != nullptr)
                context->m_legacy_graph = graph;

            for (auto& pass : *this)
            {
                status = pass->run_on_graph(graph);
                if (!status)
                    break;
            }
            return status;
        };
    };

    class Engine
    {
    public:
        Engine();
        bool run_on_graph(graph::Graph::Pointer graph, EngineContext::Pointer context = nullptr);

    protected:
        InterpreterPassManager::Pointer m_passes;
        GraphPassManager::Pointer g_passes;
        GraphVisitor::Pointer g_visitor;
    };
} // namespace nnfusion