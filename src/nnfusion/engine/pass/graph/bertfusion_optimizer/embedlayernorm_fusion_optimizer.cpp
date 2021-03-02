// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "embedlayernorm_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool EmbedLayerNormFusionOptimizer::CheckStartingNode(std::shared_ptr<nnfusion::graph::GNode> node)
{
    if (node->get_op_type() != "LayerNorm")
        return false;

    return true;
}

bool EmbedLayerNormFusionOptimizer::FindSubGraph(std::shared_ptr<GNode> starting_node,
                                                 std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);

    // find segment embedding
    std::vector<std::string> pattern_segment = {
        starting_node->get_op_type(), "Add", "GatherV2", "Parameter"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_segment;
    if (!FindPath(starting_node, pattern_segment, all_paths_segment, true) ||
        all_paths_segment.size() != 1)
    {
        // NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to segement embedding";
        return false;
    }

    auto add_before_layernorm = all_paths_segment[0][1];
    auto segment_gather = all_paths_segment[0][2];
    auto segment_id = all_paths_segment[0][3];
    auto segment_embedding = segment_gather->get_in_edge(0)->get_src();

    if (segment_embedding->get_output_shape(0).size() != 2)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to segement embedding";
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_segment[0].begin(),
                                             all_paths_segment[0].end() - 1);

    // find word embedding
    std::vector<std::string> pattern_word = {
        add_before_layernorm->get_op_type(), "Add", "GatherV2", "Parameter"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_word;
    if (!FindPath(add_before_layernorm, pattern_word, all_paths_word, true) ||
        all_paths_word.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to word embedding";
        return false;
    }

    auto add = all_paths_word[0][1];
    auto word_gather = all_paths_word[0][2];
    auto input_id = all_paths_word[0][3];
    auto word_embedding = word_gather->get_in_edge(0)->get_src();

    if (add->get_out_edges().size() != 1 || word_gather->get_out_edges().size() != 1 ||
        word_embedding->get_output_shape(0).size() != 2)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to word embedding";
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_word[0].begin(),
                                             all_paths_word[0].end() - 1);

    // find position embedding
    std::vector<std::string> pattern_position = {
        add->get_op_type(), "Broadcast", "Reshape", "GatherV2", "Slice", "Constant"};
    std::vector<std::vector<std::shared_ptr<GNode>>> all_paths_position;
    if (!FindPath(add, pattern_position, all_paths_position, true) ||
        all_paths_position.size() != 1)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to position embedding";
        return false;
    }

    auto position_gather = all_paths_position[0][3];
    auto position_embedding = position_gather->get_in_edge(0)->get_src();

    if (position_gather->get_out_edges().size() != 1 ||
        position_embedding->get_output_shape(0).size() != 2)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to position embedding";
        return false;
    }

    bertfusion_group->nodes_to_remove.insert(all_paths_position[0].begin(),
                                             all_paths_position[0].end());

    bertfusion_group->fuse_group["inputs"] = {
        input_id, segment_id, word_embedding, position_embedding, segment_embedding};
    bertfusion_group->helper_nodes.push_back(starting_node);
    bertfusion_group->edge_nodes.push_back(starting_node);

    return true;
}

bool EmbedLayerNormFusionOptimizer::FuseSubGraph(std::shared_ptr<BertFusionGroup> bertfusion_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(bertfusion_group);
    auto& fuse_group = bertfusion_group->fuse_group;
    NNFUSION_CHECK(fuse_group.find("inputs") != fuse_group.end());
    auto& inputs = fuse_group["inputs"];
    NNFUSION_CHECK(inputs.size() == 5);
    auto input_id = inputs[0];
    auto segment_id = inputs[1];
    auto word_embedding = inputs[2];
    auto position_ebedding = inputs[3];
    auto segment_ebedding = inputs[4];
    NNFUSION_CHECK(bertfusion_group->helper_nodes.size() == 1);
    auto layernorm = bertfusion_group->helper_nodes[0];
    NNFUSION_CHECK(bertfusion_group->edge_nodes.size() == 1);

    // create ebedlayernorm node

    NNFUSION_CHECK(layernorm->get_op_type() == "LayerNorm");
    auto layernorm_op = std::dynamic_pointer_cast<op::GenericOp>(layernorm->get_op_ptr());
    auto gamma = layernorm->get_in_edge(1)->get_src();
    auto beta = layernorm->get_in_edge(2)->get_src();

    std::shared_ptr<GNode> input_id_int32 = input_id;
    std::shared_ptr<GNode> segment_id_int32 = segment_id;

    if (input_id->get_output_element_type(0) != element::i32)
    {
        auto cast_op = std::make_shared<op::Convert>(element::i32);
        cast_op->set_name(input_id->get_name() + "_to_int32");
        std::shared_ptr<GNode> cast_gnode = m_graph->add_node_and_edge(cast_op, {input_id});
        input_id_int32 = cast_gnode;
    }

    if (segment_id->get_output_element_type(0) != element::i32)
    {
        auto cast_op = std::make_shared<op::Convert>(element::i32);
        cast_op->set_name(segment_id->get_name() + "_to_int32");
        std::shared_ptr<GNode> cast_gnode = m_graph->add_node_and_edge(cast_op, {segment_id});
        segment_id_int32 = cast_gnode;
    }

    auto& cfg = layernorm_op->localOpConfig.getRoot();
    float epsilon = cfg["epsilon"];

    nnfusion::op::OpConfig::any myConfig;
    myConfig["epsilon"] = epsilon;

    auto embedlayernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        "embed" + layernorm->get_name(), "EmbedLayerNorm", myConfig);

    auto embedlayernorm_gnode = m_graph->add_node_and_edge(embedlayernorm_op,
                                                           {GNodeIndex{input_id_int32, 0},
                                                            GNodeIndex{segment_id_int32, 0},
                                                            GNodeIndex{word_embedding, 0},
                                                            GNodeIndex{position_ebedding, 0},
                                                            GNodeIndex{segment_ebedding, 0},
                                                            GNodeIndex{gamma, 0},
                                                            GNodeIndex{beta, 0}});

    // replace edge
    for (auto edge_node : bertfusion_group->edge_nodes)
    {
        auto out_edges = edge_node->get_out_edges();
        for (auto out_edge : out_edges)
        {
            auto dst = out_edge->get_dst();
            int y = out_edge->get_dst_input();
            m_graph->remove_edge(out_edge);
            m_graph->add_edge(embedlayernorm_gnode, 0, dst, y);
        }
    }

    return RemoveNodes(bertfusion_group->nodes_to_remove, embedlayernorm_gnode);
}
