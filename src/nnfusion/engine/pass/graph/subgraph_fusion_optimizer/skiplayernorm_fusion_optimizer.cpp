#include "skiplayernorm_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool SkipLayerNormFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        if (gnode->get_op_type() != "LayerNorm")
            return false;
        if (gnode->get_in_edge(0)->get_src()->get_op_type() != "Add")
            return false;
        return true;
    };

    /*
Skip Layer Normalization will fuse Add + LayerNormalization into one node, and another Add if applicable
    Before fusion:

subgraph1 - Format 1:
    [Sub1]  C    [Sub2]
        \  /     /
        Add2    /
           \   /
            Add1
             |
     LayerNormalization

subgraph1 - Format 2:
      [Sub1] [Sub2]  C
         \      \   /
          \     Add2
           \    /
            Add1
             |
     LayerNormalization

subgraph2 - Format 3:
      [Sub1]   [Sub2]
         \       /
          \     /
           \   /
            Add1
             |
     LayerNormalization

After fusion:
       [Sub1]   [Sub1]
         \      /
          \    /
    SkipLayerNormalization

*/

    //subgraph 1
    {
        SubGraph::Pointer s_skiplayernorm1 = std::make_shared<SubGraph>();
        s_skiplayernorm1->name = "skiplayernorm1";
        s_skiplayernorm1->check_starting_node = check_root;

        Pattern::Pointer p_skiplayernorm1 = std::make_shared<Pattern>();
        std::vector<std::string> ops_skiplayernorm1{"LayerNorm", "Add", "Add"};
        p_skiplayernorm1->descriptions.push_back(std::make_pair(ops_skiplayernorm1, 1));
        p_skiplayernorm1->reverse_order = true;
        auto check_skiplayernorm1 = [](const PatternRecord& pr) -> bool {
            auto add2 = pr.nodes[2];
            auto broadcast = add2->get_in_edge(1)->get_src();
            if (broadcast->get_op_type() != "Broadcast")
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "failed to find skiplayernorm subgraph";
                return false;
            }
            auto bias = broadcast->get_in_edge(0)->get_src();
            if (!bias || bias->get_op_type() != "Constant")
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "failed to find skiplayernorm subgraph";
                return false;
            }
            return true;
        };
        p_skiplayernorm1->check.push_back(check_skiplayernorm1);

        s_skiplayernorm1->patterns.push_back(p_skiplayernorm1);
        m_subgraphs.push_back(s_skiplayernorm1);
    }

    //subgraph 2
    {
        SubGraph::Pointer s_skiplayernorm2 = std::make_shared<SubGraph>();
        s_skiplayernorm2->name = "skiplayernorm2";
        s_skiplayernorm2->check_starting_node = check_root;

        Pattern::Pointer p_skiplayernorm2 = std::make_shared<Pattern>();
        std::vector<std::string> ops_skiplayernorm2{"LayerNorm", "Add"};
        p_skiplayernorm2->descriptions.push_back(std::make_pair(ops_skiplayernorm2, 1));
        p_skiplayernorm2->reverse_order = true;
        s_skiplayernorm2->patterns.push_back(p_skiplayernorm2);
        m_subgraphs.push_back(s_skiplayernorm2);
    }

    return true;
}

bool SkipLayerNormFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto pr_skiplayernorm = subgraph_record->pattern_records[0];
    auto layernorm = pr_skiplayernorm->nodes[0];
    auto add1 = pr_skiplayernorm->nodes[1];
    std::shared_ptr<GNode> input, skip, broadcast, bias;
    if (subgraph_record->subgraph->name == "skiplayernorm2")
    {
        input = add1->get_in_edge(0)->get_src();
        skip = add1->get_in_edge(1)->get_src();
    }
    else if (subgraph_record->subgraph->name == "skiplayernorm1")
    {
        auto add2 = pr_skiplayernorm->nodes[2];
        if (add1->get_in_edge(0)->get_src() == add2) // format 1
        {
            input = add2->get_in_edge(0)->get_src();
            skip = add1->get_in_edge(1)->get_src();
        }
        else // format 2
        {
            input = add1->get_in_edge(0)->get_src();
            skip = add2->get_in_edge(0)->get_src();
        }

        broadcast = add2->get_in_edge(1)->get_src();
        bias = broadcast->get_in_edge(0)->get_src();
    }

    // create skiplayernorm node
    NNFUSION_CHECK(layernorm->get_op_type() == "LayerNorm");
    auto layernorm_op = std::dynamic_pointer_cast<op::GenericOp>(layernorm->get_op_ptr());
    auto gamma = layernorm->get_in_edge(1)->get_src();
    auto beta = layernorm->get_in_edge(2)->get_src();

    auto& cfg = layernorm_op->localOpConfig.getRoot();
    float epsilon = cfg["epsilon"];

    nnfusion::op::OpConfig::any myConfig;
    myConfig["epsilon"] = epsilon;

    auto skiplayernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        "skip" + layernorm->get_name(), "SkipLayerNorm", myConfig);

    std::shared_ptr<GNode> skiplayernorm_gnode;
    if (bias != nullptr)
    {
        skiplayernorm_gnode = graph->add_node_and_edge(skiplayernorm_op,
                                                       {GNodeIndex{input, 0},
                                                        GNodeIndex{skip, 0},
                                                        GNodeIndex{gamma, 0},
                                                        GNodeIndex{beta, 0},
                                                        GNodeIndex{bias, 0}});
    }
    else
    {
        skiplayernorm_gnode = graph->add_node_and_edge(
            skiplayernorm_op,
            {GNodeIndex{input, 0}, GNodeIndex{skip, 0}, GNodeIndex{gamma, 0}, GNodeIndex{beta, 0}});
    }

    std::shared_ptr<GNode> last_node = layernorm;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(skiplayernorm_gnode, 0, dst, y);
    }

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_skiplayernorm->nodes.begin(), pr_skiplayernorm->nodes.end());

    nodes_to_remove.insert(broadcast);

    return RemoveNodes(nodes_to_remove, skiplayernorm_gnode);
}