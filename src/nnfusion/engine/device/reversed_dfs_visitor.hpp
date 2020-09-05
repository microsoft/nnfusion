// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/engine/engine.hpp"

namespace nnfusion
{
    class ReversedDFSVisitor : public GraphVisitor
    {
    public:
        nnfusion::ir::Program::Pointer
            run_on_graph(shared_ptr<graph::Graph> graph,
                         EngineContext::Pointer context = nullptr) override;
    };
} // namespace nnfusion