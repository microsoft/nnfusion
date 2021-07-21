// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "subgraph_fusion_pass.hpp"
#include "subgraph_fusion_optimizer/conv_elementwise_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/subgraph_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;

DEFINE_bool(fenable_all_subgraph_fusion, false, "");
DEFINE_bool(fconv_elem_fusion, false, "");

bool SubGraphFusionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fconv_elem_fusion || FLAGS_fenable_all_subgraph_fusion)
    {
        auto optimizer = std::make_shared<ConvElemFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "ConvElemFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "ConvElemFusion Optimization Done.";
        }
    }

    return true;
}
