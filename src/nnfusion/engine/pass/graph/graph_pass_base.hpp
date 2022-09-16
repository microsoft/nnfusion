// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    class EngineContext;
    namespace pass
    {
        namespace graph
        {
            class GraphPassBase
            {
            public:
                virtual bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
                {
                    return true;
                }
                virtual bool run_on_multi_graph(
                    std::vector<std::shared_ptr<nnfusion::graph::Graph>>& graph_vec)
                {
                    return true;
                }
                void set_context(std::shared_ptr<EngineContext> context) { m_context = context; }
                std::shared_ptr<EngineContext> get_context() { return m_context; }
            private:
                std::shared_ptr<EngineContext> m_context;
            };
        }
    }
}
