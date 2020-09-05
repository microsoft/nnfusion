// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphPass
            {
            public:
                bool run(std::vector<std::shared_ptr<nnfusion::graph::Graph>>& graph_vec);
            };
        } //namespace pass
    }     // namespace graph
} // namespace nnfusion