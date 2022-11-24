#include "conv_elementwise_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool ConvElemFusionOptimizer::create_subgraphs()
{
    auto is_conv = [](std::shared_ptr<GNode> gnode) -> bool {
        return gnode->get_op_type() == "Convolution";
    };

    // conv bias relu
    Pattern::Pointer p_conv_bias_relu = std::make_shared<Pattern>();
    std::vector<std::string> ops2{"Convolution", "Add", "Relu"};
    p_conv_bias_relu->descriptions.push_back(std::make_pair(ops2, 1));

    SubGraph::Pointer s_conv_bias_relu = std::make_shared<SubGraph>();
    s_conv_bias_relu->name = "conv_bias_relu";
    s_conv_bias_relu->check_starting_node = is_conv;
    s_conv_bias_relu->patterns.push_back(p_conv_bias_relu);

    m_subgraphs.push_back(s_conv_bias_relu);

    // conv bias
    Pattern::Pointer p_conv_bias = std::make_shared<Pattern>();
    std::vector<std::string> ops1{"Convolution", "Add"};
    p_conv_bias->descriptions.push_back(std::make_pair(ops1, 1));

    SubGraph::Pointer s_conv_bias = std::make_shared<SubGraph>();
    s_conv_bias->name = "conv_bias";
    s_conv_bias->check_starting_node = is_conv;
    s_conv_bias->patterns.push_back(p_conv_bias);
    m_subgraphs.push_back(s_conv_bias);

    return true;
}

bool ConvElemFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto pr = subgraph_record->pattern_records[0];
    auto conv = pr->nodes[0];
    auto bias = pr->nodes[1];

    std::shared_ptr<GNode> relu;
    // NNFUSION_LOG(INFO) <<" =============" << subgraph_record->subgraph->name;
    if (subgraph_record->subgraph->name == "conv_bias_relu")
    {
        relu = pr->nodes[2];
    }

    std::shared_ptr<GNode> bias_input, bias_broadcast;
    int bias_input_idx;

    for (auto in_edge : bias->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != conv)
        {
            if (src->get_op_type() == "Broadcast")
            {
                bias_broadcast = src;
                bias_input = bias_broadcast->get_in_edge(0)->get_src();
                bias_input_idx = 0;
            }
            else
            {
                bias_input = src;
                bias_input_idx = in_edge->get_src_output();
            }

            // NNFUSION_LOG(INFO) << "bias_input_idx: " << bias_input_idx;
            // NNFUSION_LOG(INFO) << bias_input->get_op_type();
            break;
        }
    }
    NNFUSION_CHECK_NOT_NULLPTR(bias_input);

    GNodeIndexVector conv_inputs;
    for (size_t i = 0; i < conv->get_in_edges().size(); i++)
    {
        auto edge = conv->get_in_edge(i);
        auto src = edge->get_src();
        auto src_idx = edge->get_src_output();
        conv_inputs.push_back(GNodeIndex{src, src_idx});
    }

    conv_inputs.push_back(GNodeIndex{bias_input, bias_input_idx});

    auto conv_op = std::dynamic_pointer_cast<op::Convolution>(conv->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(conv_op);
    if (relu)
        conv_op->set_activation("relu");

    auto new_conv = graph->add_node_and_edge(conv_op, conv_inputs);

    std::shared_ptr<GNode> last_node;
    if (relu)
        last_node = relu;
    else
        last_node = bias;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(new_conv, 0, dst, y);
    }
    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert({conv, bias, bias_broadcast});
    if (relu)
        nodes_to_remove.insert(relu);
    return RemoveNodes(nodes_to_remove, new_conv);
}