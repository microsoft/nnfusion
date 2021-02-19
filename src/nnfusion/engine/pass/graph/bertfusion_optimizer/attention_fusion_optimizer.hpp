// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "bertfusion_optimizer.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AttentionFusionOptimizer : public BertFusionOptimizer
            {
            public:
                AttentionFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> graph)
                    : BertFusionOptimizer(graph)

                {
                }

            private:
                bool CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node) override;
                bool FindSubGraph(std::shared_ptr<nnfusion::graph::GNode> starting_node,
                                  std::shared_ptr<BertFusionGroup> bertfusion_group) override;
                bool FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group) override;
                std::shared_ptr<nnfusion::graph::GNode>
                    MergeQkvWeights(std::shared_ptr<nnfusion::graph::GNode> q_weight,
                                    std::shared_ptr<nnfusion::graph::GNode> k_weight,
                                    std::shared_ptr<nnfusion::graph::GNode> v_weight,
                                    size_t hidden_size,
                                    bool is_matmul);
                void MergeWeights(const char* q_weight_dptr,
                                  const char* k_weight_dptr,
                                  const char* v_weight_dptr,
                                  size_t step,
                                  std::vector<char>& qkv_weight_data);
                void MergeMatMulWeights(const char* q_weight_dptr,
                                        const char* k_weight_dptr,
                                        const char* v_weight_dptr,
                                        size_t step,
                                        std::vector<char>& qkv_weight_data);
                std::shared_ptr<nnfusion::graph::GNode>
                    GetorCreateMaskIndex(std::shared_ptr<nnfusion::graph::GNode> mask_input);

                std::map<std::string, std::shared_ptr<nnfusion::graph::GNode>> mask_index_map;
                std::map<std::string, std::shared_ptr<nnfusion::graph::GNode>> qkv_weight_map;
            };

        } // namespace graph
    }     // namespace pass
} // namespace nnfusion
