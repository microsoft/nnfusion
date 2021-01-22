// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "bertfusion_pass.hpp"
#include "bertfusion_optimizer/attention_fusion_optimizer.hpp"
#include "bertfusion_optimizer/bertfusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;

DEFINE_bool(fattention_fusion, false, "");

bool BertFusionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fattention_fusion)
    {
        auto optimizer = std::make_shared<AttentionFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "BertAttentionFusion Optimization failed.";
        }
    }

    return true;
}
