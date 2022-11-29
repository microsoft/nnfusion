#include "layernorm_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool LayerNormFusionOptimizer::create_subgraphs()
{
    /*
+---------------------+
|                     |
|                     v
X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                      |                                               ^
                      |                                               |
                      +-----------------------------------------------+
*/
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        if (gnode->get_out_edges().size() != 2)
            return false;
        int sum_count = 0;
        int sub_count = 0;
        for (auto out_edge : gnode->get_out_edges())
        {
            auto dst = out_edge->get_dst();
            if (dst->get_op_type() == "Sum")
            {
                sum_count += 1;
            }
            else if (dst->get_op_type() == "Subtract")
            {
                sub_count += 1;
            }
        }

        if (sum_count != 1 || sub_count != 1)
            return false;

        return true;
    };
    SubGraph::Pointer s_layernorm = std::make_shared<SubGraph>();
    s_layernorm->name = "layernorm";
    s_layernorm->check_starting_node = check_root;
    std::vector<std::string> ops_reduce_mean{"AnyOp", "Sum", "Divide", "Reshape"};
    auto check_reduce_mean = [](const PatternRecord& pr) -> bool {
        auto divide = pr.nodes[2];
        for (auto in_edge : divide->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src->get_op_type() == "Broadcast")
            {
                auto src_input = src->get_in_edge(0)->get_src();
                // todo: remove reshape
                if (src_input->is_constant() ||
                    (src_input->get_op_type() == "Reshape" &&
                     src_input->get_in_edge(0)->get_src()->is_constant()))
                    return true;
            }
        }
        return false;
    };

    // reducemean1
    {
        Pattern::Pointer p_reduce_mean1 = std::make_shared<Pattern>();
        p_reduce_mean1->descriptions.push_back(std::make_pair(ops_reduce_mean, 3));
        p_reduce_mean1->reverse_order = false;
        p_reduce_mean1->check.push_back(check_reduce_mean);
        auto check_reduce_mean1 = [](const PatternRecord& pr) -> bool {

            // check axis value is available
            auto reducemean1_sum = pr.nodes[1];
            auto reducemean1_op = std::dynamic_pointer_cast<op::Sum>(reducemean1_sum->get_op_ptr());
            std::vector<size_t> axis_vec;
            for (auto axis : reducemean1_op->get_reduction_axes())
            {
                axis_vec.push_back(axis);
            }

            if (axis_vec.size() != 1)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find axis";
                return false;
            }

            return true;
        };
        p_reduce_mean1->check.push_back(check_reduce_mean1);

        s_layernorm->patterns.push_back(p_reduce_mean1);
    }

    // path to reducemean2
    {
        Pattern::Pointer p_to_reducemean2 = std::make_shared<Pattern>();
        std::vector<std::string> ops_to_reducemean2_1{
            "AnyOp", "Reshape", "Broadcast", "Subtract", "Power"};
        std::vector<std::string> ops_to_reducemean2_2{
            "AnyOp", "Reshape", "Broadcast", "Subtract", "Convert", "Power"};
        p_to_reducemean2->descriptions.push_back(std::make_pair(ops_to_reducemean2_1, 4));
        p_to_reducemean2->descriptions.push_back(std::make_pair(ops_to_reducemean2_2, 5));
        p_to_reducemean2->reverse_order = false;
        s_layernorm->patterns.push_back(p_to_reducemean2);
    }

    // reducemean2
    {
        Pattern::Pointer p_reduce_mean2 = std::make_shared<Pattern>();
        p_reduce_mean2->descriptions.push_back(std::make_pair(ops_reduce_mean, 3));
        p_reduce_mean2->reverse_order = false;
        p_reduce_mean2->check.push_back(check_reduce_mean);
        s_layernorm->patterns.push_back(p_reduce_mean2);
    }

    // path to last add
    {
        Pattern::Pointer p_to_last_add = std::make_shared<Pattern>();
        std::vector<std::string> ops_to_last_add{
            "AnyOp", "Add", "Sqrt", "Reshape", "Broadcast", "Divide", "Multiply", "Add"};

        p_to_last_add->descriptions.push_back(std::make_pair(ops_to_last_add, 1));
        p_to_last_add->reverse_order = false;
        auto check_to_last_add = [](const PatternRecord& pr) -> bool {
            auto add2 = pr.nodes[1];
            auto multiply = pr.nodes[6];
            auto last_add = pr.nodes.back();
            std::shared_ptr<GNode> epsilon, scale, bias, epsilon_broadcast, epsilon_reshape,
                scale_broadcast, bias_broadcast;
            for (auto in_edge : add2->get_in_edges())
            {
                auto src = in_edge->get_src();
                if (src->get_op_type() == "Broadcast")
                {
                    epsilon_broadcast = src;
                    epsilon_reshape = src->get_in_edge(0)->get_src();
                    if (epsilon_reshape->get_op_type() != "Reshape")
                    {
                        NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
                        return false;
                    }
                    epsilon = epsilon_reshape->get_in_edge(0)->get_src();
                }
            }
            if (!epsilon_broadcast || !epsilon_reshape || !epsilon)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to  to the last add";
                return false;
            }

            std::vector<float> epsilon_value;
            bool status = nnfusion::frontend::GetValueFromNGraphOp<float>(epsilon, &epsilon_value);
            if (!status || epsilon_value.size() != 1)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
                return false;
            }

            for (auto in_edge : multiply->get_in_edges())
            {
                auto src = in_edge->get_src();
                if (src->get_op_type() == "Broadcast")
                {
                    scale_broadcast = src;
                    auto cur = src->get_in_edge(0)->get_src();
                    // todo : remove reshape
                    if (cur->is_constant())
                        scale = cur;
                    else if (cur->get_op_type() == "Reshape" &&
                             cur->get_in_edge(0)->get_src()->is_constant())
                        scale = cur->get_in_edge(0)->get_src();
                }
            }
            if (!scale_broadcast || !scale)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to  to the last add";
                return false;
            }

            for (auto in_edge : last_add->get_in_edges())
            {
                auto src = in_edge->get_src();
                if (src->get_op_type() == "Broadcast")
                {
                    bias_broadcast = src;
                    bias = src->get_in_edge(0)->get_src();
                }
            }

            if (!bias_broadcast || !bias)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find path to  to the last add";
                return false;
            }

            return true;
        };
        p_to_last_add->check.push_back(check_to_last_add);
        s_layernorm->patterns.push_back(p_to_last_add);
    }

    auto check_layernorm1 = [](const SubGraphRecord& sr) -> bool {

        // check subtract after the first reducemean is the child of starting node
        auto pr_to_reducemean2 = sr.pattern_records[1];
        auto subtract = pr_to_reducemean2->nodes[3];

        for (auto in_edge : subtract->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src == sr.get_starting_node())
                return true;
        }
        return false;
    };
    s_layernorm->check.push_back(check_layernorm1);

    auto check_layernorm2 = [](const SubGraphRecord& sr) -> bool {

        // check div is the child of subtract
        auto pr_to_reducemean2 = sr.pattern_records[1];
        auto subtract = pr_to_reducemean2->nodes[3];

        auto pr_to_last_add = sr.pattern_records[3];
        auto divide = pr_to_last_add->nodes[5];

        for (auto in_edge : divide->get_in_edges())
        {
            auto src = in_edge->get_src();
            if (src == subtract)
                return true;
        }
        return false;
    };
    s_layernorm->check.push_back(check_layernorm2);

    m_subgraphs.push_back(s_layernorm);
    return true;
}

bool LayerNormFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto input_node = subgraph_record->get_starting_node();
    auto pr_reducemean1 = subgraph_record->pattern_records[0];
    auto reducemean1_divide = pr_reducemean1->nodes[2];
    std::shared_ptr<GNode> reducemean1_broadcast_before_div, reducemean1_const_before_div;
    for (auto in_edge : reducemean1_divide->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            reducemean1_broadcast_before_div = src;
            reducemean1_const_before_div =
                reducemean1_broadcast_before_div->get_in_edge(0)->get_src();
        }
    }
    auto reducemean1_sum = pr_reducemean1->nodes[1];
    auto pr_to_reducemean2 = subgraph_record->pattern_records[1];
    auto pr_reducemean2 = subgraph_record->pattern_records[2];
    auto reducemean2_divide = pr_reducemean2->nodes[2];
    std::shared_ptr<GNode> reducemean2_broadcast_before_div, reducemean2_const_before_div;
    for (auto in_edge : reducemean2_divide->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            reducemean2_broadcast_before_div = src;
            reducemean2_const_before_div =
                reducemean2_broadcast_before_div->get_in_edge(0)->get_src();
        }
    }
    auto pr_to_last_add = subgraph_record->pattern_records[3];
    auto add2 = pr_to_last_add->nodes[1];
    auto multiply = pr_to_last_add->nodes[6];
    auto last_add = pr_to_last_add->nodes.back();
    std::shared_ptr<GNode> epsilon, scale, bias, epsilon_broadcast, epsilon_reshape,
        scale_broadcast, bias_broadcast;
    for (auto in_edge : add2->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            epsilon_broadcast = src;
            epsilon_reshape = src->get_in_edge(0)->get_src();
            epsilon = epsilon_reshape->get_in_edge(0)->get_src();
            break;
        }
    }
    for (auto in_edge : multiply->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            scale_broadcast = src;
            // todo: remove reshape
            if (src->get_in_edge(0)->get_src()->is_constant())
                scale = src->get_in_edge(0)->get_src();
            else if (src->get_in_edge(0)->get_src()->get_op_type() == "Reshape")
                scale = src->get_in_edge(0)->get_src()->get_in_edge(0)->get_src();
        }
    }

    for (auto in_edge : last_add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src->get_op_type() == "Broadcast")
        {
            bias_broadcast = src;
            bias = src->get_in_edge(0)->get_src();
        }
    }

    // create layernorm node
    auto reducemean1_op = std::dynamic_pointer_cast<op::Sum>(reducemean1_sum->get_op_ptr());
    std::vector<size_t> axis_vec;
    for (auto axis : reducemean1_op->get_reduction_axes())
    {
        axis_vec.push_back(axis);
    }

    // if (axis_vec.size() != 1)
    // {
    //     NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find axis";
    //     return false;
    // }

    std::vector<float> epsilon_value;
    bool status = nnfusion::frontend::GetValueFromNGraphOp<float>(epsilon, &epsilon_value);
    // if (!status || epsilon_value.size() != 1)
    // {
    //     NNFUSION_LOG(NNFUSION_WARNING) << "Failed to find epsilon";
    //     return false;
    // }

    // todo: move the check to subgraph definition
    {
        auto input_shape_0 = input_node->get_output_shape(0);
        auto input_shape_1 = scale->get_output_shape(0);
        auto input_shape_2 = bias->get_output_shape(0);
        size_t axis = axis_vec[0];
        if (input_shape_1 != input_shape_2 || input_shape_1.size() != input_shape_0.size() - axis ||
            !std::equal(input_shape_1.begin(), input_shape_1.end(), input_shape_0.begin() + axis))
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Invalid LayerNorm op.";
            return false;
        }
    }

    nnfusion::op::OpConfig::any myConfig;
    myConfig["axis"] = axis_vec[0];
    // NNFUSION_LOG(INFO) << axis_vec[0] << ", " << input_node->get_output_shape(0).size() << ", " << scale->get_output_shape(0).size();
    myConfig["epsilon"] = epsilon_value[0];
    auto layernorm_op = std::make_shared<nnfusion::op::GenericOp>(
        input_node->get_name() + "_layernorm", "LayerNorm", myConfig);
    auto layernorm_gnode = graph->add_node_and_edge(
        layernorm_op, {GNodeIndex{input_node, 0}, GNodeIndex{scale, 0}, GNodeIndex{bias, 0}});

    std::shared_ptr<GNode> last_node = last_add;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(layernorm_gnode, 0, dst, y);
    }

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_reducemean1->nodes.begin() + 1, pr_reducemean1->nodes.end());
    nodes_to_remove.insert(pr_to_reducemean2->nodes.begin(), pr_to_reducemean2->nodes.end());
    nodes_to_remove.insert(pr_reducemean2->nodes.begin() + 1, pr_reducemean2->nodes.end());
    nodes_to_remove.insert(pr_to_last_add->nodes.begin(), pr_to_last_add->nodes.end());
    nodes_to_remove.insert({
        reducemean1_broadcast_before_div,
        reducemean1_const_before_div,
        reducemean2_broadcast_before_div,
        reducemean2_const_before_div,
    });
    nodes_to_remove.insert(
        {epsilon_broadcast, scale_broadcast, bias_broadcast, epsilon_reshape, epsilon});

    return RemoveNodes(nodes_to_remove, layernorm_gnode);
}