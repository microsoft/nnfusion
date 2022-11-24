// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class DumpOp : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        }
    }
} // namespace nnfusion