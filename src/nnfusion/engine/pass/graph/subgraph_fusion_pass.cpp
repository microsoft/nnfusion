// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "subgraph_fusion_pass.hpp"
#include "subgraph_fusion_optimizer/attention_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/conv_elementwise_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/embed_layernorm_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/gelu_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/layernorm_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/matmuladd_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/mem_eff_attn_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/skiplayernorm_fusion_optimizer.hpp"
#include "subgraph_fusion_optimizer/subgraph_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;

DEFINE_bool(fenable_all_subgraph_fusion, false, "");
DEFINE_bool(fenable_all_bert_fusion, false, "");
DEFINE_bool(fconv_elem_fusion, false, "");
DEFINE_bool(fattention_fusion, false, "");
DEFINE_bool(flayernorm_fusion, false, "");
DEFINE_bool(fembedlayernorm_fusion, false, "");
DEFINE_bool(fskiplayernorm_fusion, false, "");
DEFINE_bool(fgelu_fusion, false, "");
DEFINE_bool(fmatmuladd_fusion, false, "");
DEFINE_bool(fmem_eff_attn_fusion, false, "");

bool SubGraphFusionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool enable_layernorm_fusion = (FLAGS_flayernorm_fusion || FLAGS_fenable_all_bert_fusion ||
                                    FLAGS_fenable_all_subgraph_fusion);
    bool enable_attention_fusion = (FLAGS_fattention_fusion || FLAGS_fenable_all_bert_fusion ||
                                    FLAGS_fenable_all_subgraph_fusion);
    bool enable_embedlayernorm_fusion =
        (FLAGS_fembedlayernorm_fusion || FLAGS_fenable_all_bert_fusion ||
         FLAGS_fenable_all_subgraph_fusion);
    bool enable_skiplayernorm_fusion =
        (FLAGS_fskiplayernorm_fusion || FLAGS_fenable_all_bert_fusion ||
         FLAGS_fenable_all_subgraph_fusion);
    bool enable_gelu_fusion =
        (FLAGS_fgelu_fusion || FLAGS_fenable_all_bert_fusion || FLAGS_fenable_all_subgraph_fusion);
    bool enable_matmuladd_fusion = (FLAGS_fmatmuladd_fusion || FLAGS_fenable_all_subgraph_fusion);
    bool enable_conv_elem_fusion = (FLAGS_fconv_elem_fusion || FLAGS_fenable_all_subgraph_fusion);

    if (enable_embedlayernorm_fusion || enable_skiplayernorm_fusion)
        enable_layernorm_fusion = true;

    if (enable_layernorm_fusion)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "LayerNormFusion Optimization begin.";
        auto optimizer = std::make_shared<LayerNormFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "LayerNormFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "LayerNormFusion Optimization Done.";
        }
    }

    if (enable_attention_fusion)
    {
        auto optimizer = std::make_shared<AttentionFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "AttentionFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "AttentionFusion Optimization Done.";
        }
    }

    if (FLAGS_fmem_eff_attn_fusion)
    {
        auto optimizer = std::make_shared<MemEffAttnFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "MemEffAttentionFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "MemEffAttentionFusion Optimization Done.";
        }
    }

    if (enable_embedlayernorm_fusion)
    {
        auto optimizer = std::make_shared<EmbedLayerNormFusionOptimizer>(graph);
        if (!optimizer->Optimize())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "EmbedLayerNormFusion Optimization failed.";
        }
        else
        {
            NNFUSION_LOG(INFO) << "EmbedLayerNormFusion Optimization Done.";
        }
    }

    if (enable_skiplayernorm_fusion)
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

    if (enable_gelu_fusion)
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

    if (enable_matmuladd_fusion)
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

    if (enable_conv_elem_fusion)
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
