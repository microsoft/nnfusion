#include "gelu_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool GeluFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        if (gnode->get_out_edges().size() != 2)
            return false;
        std::shared_ptr<GNode> divide;
        int mul_count = 0;
        int div_count = 0;
        for (auto edge : gnode->get_out_edges())
        {
            auto dst = edge->get_dst();
            if (dst->get_op_type() == "Multiply")
            {
                mul_count += 1;
            }
            else if (dst->get_op_type() == "Divide")
            {
                div_count += 1;
                divide = dst;
            }
        }

        if (mul_count != 1 || div_count != 1)
            return false;

        return true;
    };

    SubGraph::Pointer s_gelu = std::make_shared<SubGraph>();
    s_gelu->name = "gelu";
    s_gelu->check_starting_node = check_root;

    /*
    subgraph 1
                   +-------Mul(0.5)---------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul ==>
                          (B=1.4142...)        (1)
    

    Subgraph 2
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)
    After Fusion:
                [root]--> Gelu ==>
*/

    // mul path
    {
        Pattern::Pointer p_mul = std::make_shared<Pattern>();
        std::vector<std::string> ops_mul{"AnyOp", "Multiply", "Multiply"};
        // for subgraph 1
        p_mul->descriptions.push_back(std::make_pair(ops_mul, 2));
        // for subgraph 2
        p_mul->descriptions.push_back(std::make_pair(ops_mul, 1));
        p_mul->reverse_order = false;
        s_gelu->patterns.push_back(p_mul);
    }

    // div path
    {
        Pattern::Pointer p_div = std::make_shared<Pattern>();
        std::vector<std::string> ops_div{"Multiply", "Add", "Erf", "Divide"};
        p_div->descriptions.push_back(std::make_pair(ops_div, 1));
        p_div->reverse_order = true;
        auto check_div = [](const PatternRecord& pr) -> bool {
            auto divide = pr.nodes[3];
            auto broadcast_before_div = divide->get_in_edge(1)->get_src();
            if (broadcast_before_div->get_op_type() != "Broadcast")
            {
                // NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2)";
                return false;
            }

            auto const_before_div = broadcast_before_div->get_in_edge(0)->get_src();
            std::vector<float> const_value;
            bool status =
                nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_div, &const_value);
            if (!status || const_value.size() != 1)
            {
                // NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2)";
                return false;
            }

            float approximate_sqrt2 = 1.4142099618911743f;
            float diff = std::abs(const_value[0] - approximate_sqrt2);
            const float atol = 1e-8f;
            const float rtol = 1e-3f;
            if (diff > (atol + rtol * std::abs(approximate_sqrt2)))
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find sqrt(2), got " << const_value[0];
                return false;
            }

            auto erf = pr.nodes[2];
            auto add = pr.nodes[1];

            std::shared_ptr<GNode> broadcast_before_add, const_before_add;
            for (auto in_edge : add->get_in_edges())
            {
                auto src = in_edge->get_src();
                if (src != erf)
                {
                    broadcast_before_add = src;
                    if (broadcast_before_add->get_op_type() != "Broadcast")
                    {
                        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                        return false;
                    }
                    const_before_add = broadcast_before_add->get_in_edge(0)->get_src();
                    const_value.clear();
                    status = nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_add,
                                                                             &const_value);
                    if (!status || const_value.size() != 1 || const_value[0] != 1)
                    {
                        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
                        return false;
                    }
                    break;
                }
            }
            return true;
        };
        p_div->check.push_back(check_div);
        s_gelu->patterns.push_back(p_div);
    }

    auto check_gelu1 = [](const SubGraphRecord& sr) -> bool {

        // check divide is the chid of starting node
        auto pr_div = sr.pattern_records[1];
        auto div = pr_div->nodes[3];

        for (auto in_edge : div->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src == sr.get_starting_node())
                return true;
        }
        return false;
    };
    s_gelu->check.push_back(check_gelu1);

    auto check_gelu2 = [](const SubGraphRecord& sr) -> bool {
        // check mul 0.5
        std::shared_ptr<GNode> mul, broadcast_before_mul, const_before_mul;
        auto pr_mul = sr.pattern_records[0];
        if (pr_mul->get_pattern_description_idx() == 0)
        {
            mul = pr_mul->nodes[1];
        }
        if (pr_mul->get_pattern_description_idx() == 1)
        {
            mul = pr_mul->nodes[2];
        }

        if (!mul)
            return false;

        for (auto in_edge : mul->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src->get_op_type() == "Broadcast")
            {
                broadcast_before_mul = src;
                break;
            }
        }

        if (!broadcast_before_mul)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
            return false;
        }

        std::vector<float> const_value;
        const_before_mul = broadcast_before_mul->get_in_edge(0)->get_src();
        const_value.clear();
        bool status =
            nnfusion::frontend::GetValueFromNGraphOp<float>(const_before_mul, &const_value);
        if (!status || const_value.size() != 1 || const_value[0] != 0.5)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find gelu subgraph";
            return false;
        }

        return true;
    };
    s_gelu->check.push_back(check_gelu2);

    m_subgraphs.push_back(s_gelu);
    return true;
}

bool GeluFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto input = subgraph_record->get_starting_node();
    auto pr_mul = subgraph_record->pattern_records[0];
    auto pr_div = subgraph_record->pattern_records[1];
    auto mul2 = pr_mul->nodes[2];
    auto div = pr_div->nodes[3];
    auto broadcast_before_div = div->get_in_edge(1)->get_src();
    auto const_before_div = broadcast_before_div->get_in_edge(0)->get_src();
    auto add = pr_div->nodes[1];
    auto erf = pr_div->nodes[2];
    std::shared_ptr<GNode> broadcast_before_add, const_before_add;
    for (auto in_edge : add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != erf)
        {
            broadcast_before_add = src;
            const_before_add = broadcast_before_add->get_in_edge(0)->get_src();
            break;
        }
    }

    std::shared_ptr<GNode> mul, broadcast_before_mul, const_before_mul;
    if (pr_mul->get_pattern_description_idx() == 0)
    {
        mul = pr_mul->nodes[1];
    }
    if (pr_mul->get_pattern_description_idx() == 1)
    {
        mul = pr_mul->nodes[2];
    }

    for (auto in_edge : mul->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            broadcast_before_mul = src;
            break;
        }
    }
    const_before_mul = broadcast_before_mul->get_in_edge(0)->get_src();

    // create gelu node
    auto gelu_op = std::make_shared<nnfusion::op::Gelu>();
    auto gelu_gnode = graph->add_node_and_edge(gelu_op, {GNodeIndex{input, 0}});

    std::shared_ptr<GNode> last_node = mul2;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(gelu_gnode, 0, dst, y);
    }

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_div->nodes.begin(), pr_div->nodes.end());
    nodes_to_remove.insert(pr_mul->nodes.begin() + 1, pr_mul->nodes.end());
    nodes_to_remove.insert({broadcast_before_div,
                            const_before_div,
                            broadcast_before_add,
                            const_before_add,
                            broadcast_before_mul,
                            const_before_mul});

    return RemoveNodes(nodes_to_remove, gelu_gnode);
}