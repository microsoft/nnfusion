#include "embed_layernorm_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool EmbedLayerNormFusionOptimizer::create_subgraphs()
{
    SubGraph::Pointer s_embed_layernorm = std::make_shared<SubGraph>();
    s_embed_layernorm->name = "embed_layernorm";
    auto is_layernorm = [](std::shared_ptr<GNode> gnode) -> bool {
        return gnode->get_op_type() == "Convolution";
    };
    s_embed_layernorm->check_starting_node = is_layernorm;

    // segment embedding
    {
        Pattern::Pointer p_segment = std::make_shared<Pattern>();
        std::vector<std::string> ops_segment{"LayerNorm", "Add", "GatherV2", "Parameter"};
        p_segment->descriptions.push_back(std::make_pair(ops_segment, 1));
        p_segment->reverse_order = true;
        auto check_seg_ebed = [](const PatternRecord& pr) -> bool {
            auto segment_gather = pr.nodes[2];
            auto segment_embedding = segment_gather->get_in_edge(0)->get_src();
            return segment_embedding->get_output_shape(0).size() == 2;
        };
        p_segment->check.push_back(check_seg_ebed);
        s_embed_layernorm->patterns.push_back(p_segment);
    }

    // word embedding
    {
        Pattern::Pointer p_word = std::make_shared<Pattern>();
        std::vector<std::string> ops_word{"Add", "Add", "GatherV2", "Parameter"};
        p_word->descriptions.push_back(std::make_pair(ops_word, 1));
        p_word->reverse_order = true;
        auto check_word_ebed = [](const PatternRecord& pr) -> bool {
            auto add = pr.nodes[1];
            auto word_gather = pr.nodes[2];
            auto word_embedding = word_gather->get_in_edge(0)->get_src();

            return (add->get_out_edges().size() == 1 && word_gather->get_out_edges().size() == 1 &&
                    word_embedding->get_output_shape(0).size() == 2);
        };
        p_word->check.push_back(check_word_ebed);
        s_embed_layernorm->patterns.push_back(p_word);
    }

    // position embedding
    {
        Pattern::Pointer p_position = std::make_shared<Pattern>();
        std::vector<std::string> ops_position{
            "Add", "Broadcast", "Reshape", "GatherV2", "Slice", "Constant"};
        p_position->descriptions.push_back(std::make_pair(ops_position, 1));
        p_position->reverse_order = true;
        auto check_position_ebed = [](const PatternRecord& pr) -> bool {
            auto position_gather = pr.nodes[3];
            auto position_embedding = position_gather->get_in_edge(0)->get_src();
            return (position_gather->get_out_edges().size() == 1 &&
                    position_embedding->get_output_shape(0).size() == 2);
        };
        p_position->check.push_back(check_position_ebed);
        s_embed_layernorm->patterns.push_back(p_position);
    }

    m_subgraphs.push_back(s_embed_layernorm);
    return true;
}

bool EmbedLayerNormFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto pr_segment = subgraph_record->pattern_records[0];
    auto pr_word = subgraph_record->pattern_records[1];
    auto pr_position = subgraph_record->pattern_records[2];
    auto input_id = pr_word->nodes[3];
    auto word_gather = pr_word->nodes[2];
    auto word_embedding = word_gather->get_in_edge(0)->get_src();
    auto segment_id = pr_segment->nodes[3];
    auto segment_gather = pr_segment->nodes[2];
    auto segment_embedding = segment_gather->get_in_edge(0)->get_src();
    auto position_gather = pr_position->nodes[3];
    auto position_embedding = position_gather->get_in_edge(0)->get_src();
    auto layernorm = subgraph_record->get_starting_node();

    // create ebedlayernorm node
    auto layernorm_op = std::dynamic_pointer_cast<op::GenericOp>(layernorm->get_op_ptr());
    auto gamma = layernorm->get_in_edge(1)->get_src();
    auto beta = layernorm->get_in_edge(2)->get_src();

    std::shared_ptr<GNode> input_id_int32 = input_id;
    std::shared_ptr<GNode> segment_id_int32 = segment_id;

    if (input_id->get_output_element_type(0) != element::i32)
    {
        auto cast_op = std::make_shared<op::Convert>(element::i32);
        cast_op->set_name(input_id->get_name() + "_to_int32");
        std::shared_ptr<GNode> cast_gnode = graph->add_node_and_edge(cast_op, {input_id});
        input_id_int32 = cast_gnode;
    }

    if (segment_id->get_output_element_type(0) != element::i32)
    {
        auto cast_op = std::make_shared<op::Convert>(element::i32);
        cast_op->set_name(segment_id->get_name() + "_to_int32");
        std::shared_ptr<GNode> cast_gnode = graph->add_node_and_edge(cast_op, {segment_id});
        segment_id_int32 = cast_gnode;
    }

    auto& cfg = layernorm_op->localOpConfig.getRoot();
    float epsilon = cfg["epsilon"];

    nnfusion::op::OpConfig::any myConfig;
    myConfig["epsilon"] = epsilon;

    auto embedlayernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        "embed" + layernorm->get_name(), "EmbedLayerNorm", myConfig);

    auto embedlayernorm_gnode = graph->add_node_and_edge(embedlayernorm_op,
                                                         {GNodeIndex{input_id_int32, 0},
                                                          GNodeIndex{segment_id_int32, 0},
                                                          GNodeIndex{word_embedding, 0},
                                                          GNodeIndex{position_embedding, 0},
                                                          GNodeIndex{segment_embedding, 0},
                                                          GNodeIndex{gamma, 0},
                                                          GNodeIndex{beta, 0}});

    std::shared_ptr<GNode> last_node = layernorm;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(embedlayernorm_gnode, 0, dst, y);
    }
    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_segment->nodes.begin(), pr_segment->nodes.end() - 1);
    nodes_to_remove.insert(pr_word->nodes.begin(), pr_word->nodes.end() - 1);
    nodes_to_remove.insert(pr_position->nodes.begin(), pr_position->nodes.end());

    return RemoveNodes(nodes_to_remove, embedlayernorm_gnode);
}