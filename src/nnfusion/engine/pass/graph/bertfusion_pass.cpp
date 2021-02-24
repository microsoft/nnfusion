// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "bertfusion_pass.hpp"
#include "bertfusion_optimizer/attention_fusion_optimizer.hpp"
#include "bertfusion_optimizer/bertfusion_optimizer.hpp"
#include "bertfusion_optimizer/embedlayernorm_fusion_optimizer.hpp"
#include "bertfusion_optimizer/gelu_fusion_optimizer.hpp"
#include "bertfusion_optimizer/layernorm_fusion_optimizer.hpp"
#include "bertfusion_optimizer/matmuladd_fusion_optimizer.hpp"
#include "bertfusion_optimizer/skiplayernorm_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;

DEFINE_bool(fattention_fusion, false, "");
DEFINE_bool(flayernorm_fusion, false, "");
DEFINE_bool(fembedlayernorm_fusion, false, "");
DEFINE_bool(fskiplayernorm_fusion, false, "");
DEFINE_bool(fgelu_fusion, false, "");
DEFINE_bool(fmatmuladd_fusion, false, "");
DEFINE_bool(fenable_all_bert_fusion, false, "");

bool BertFusionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_flayernorm_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<LayerNormFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "BertLayerNormFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "BertLayerNormFusion Optimization Done.";
        }
    }

    if (FLAGS_fattention_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<AttentionFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "BertAttentionFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "BertAttentionFusion Optimization Done.";
        }
    }

    if (FLAGS_fembedlayernorm_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<EmbedLayerNormFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "BertEmbedLayerNormFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "BertEmbedLayerNormFusion Optimization Done.";
        }
    }

    if (FLAGS_fskiplayernorm_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<SkipLayerNormFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "SkipLayerNormFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "SkipLayerNormFusion Optimization Done.";
        }
    }

    if (FLAGS_fgelu_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<GeluFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "GeluFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "GeluFusion Optimization Done.";
        }
    }

    if (FLAGS_fmatmuladd_fusion || FLAGS_fenable_all_bert_fusion)
    {
        auto optimizer = std::make_shared<MatMulAddFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "MatMulAddFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "MatMulAddFusion Optimization Done.";
        }
    }
    return true;
}
