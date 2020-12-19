// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include "graph_pass_base.hpp"
using namespace nnfusion::graph;
DEFINE_string(fnnfusion_graph_path, "./nnfusion_graph.pb", "path to save nnfusion graph file.");
DEFINE_bool(fenable_export_graph, false, "enable exporting graph.");
namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphSerializationPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
                {
                    if (FLAGS_fenable_export_graph)
                        graph->serialize_to_file(FLAGS_fnnfusion_graph_path);
                    return true;
                }
            };
        }
    }
}
