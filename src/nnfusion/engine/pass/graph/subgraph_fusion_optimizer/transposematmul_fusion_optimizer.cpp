#include "transposematmul_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool TransposeMatMulFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        if (gnode->get_op_type() != "Reshape")
            return false;
        if (gnode->get_out_edges().size() != 1)
            return false;
        auto reshape_op = std::dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr());
        if (reshape_op == nullptr)
            return false;
        if (!(reshape_op->get_is_transpose()))
            return false;
        return true;
    };

    SubGraph::Pointer s_transmatmul = std::make_shared<SubGraph>();
    s_transmatmul->name = "transposematmul";
    s_transmatmul->check_starting_node = check_root;

    {
        Pattern::Pointer p_transmatmul = std::make_shared<Pattern>();
        std::vector<std::string> ops_mul{"Reshape", "Dot"};
        p_transmatmul->descriptions.push_back(std::make_pair(ops_mul, 1));
        p_transmatmul->reverse_order = false;
        auto check_transmatmul = [](const PatternRecord& pr) -> bool {
            auto transpose = pr.nodes[0];
            auto matmul = pr.nodes[1];
            // hardcode check for bloom model
            if (matmul->get_in_edge(1)->get_src() != transpose)
                return false;
            return true;
            // return (matmul->get_output_element_type(0) == add->get_output_element_type(0));
        };
        p_transmatmul->check.push_back(check_transmatmul);

        s_transmatmul->patterns.push_back(p_transmatmul);
    }

    m_subgraphs.push_back(s_transmatmul);
    return true;
}

bool TransposeMatMulFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto pr_transmatmul = subgraph_record->pattern_records[0];
    auto transpose = pr_transmatmul->nodes[0];
    auto matmul = pr_transmatmul->nodes[1];
    auto matmul_a = matmul->get_in_edge(0)->get_src();
    auto matmul_b = matmul->get_in_edge(1)->get_src();

    std::shared_ptr<GNode> nodeC;
    for (auto in_edge : matmul->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != transpose)
        {
            nodeC = src;
            break;
        }
    }

    // create matmuladd node
    auto cur_matmul_op = std::dynamic_pointer_cast<op::Dot>(matmul->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(cur_matmul_op);
    bool trans_A = cur_matmul_op->get_transpose_A();
    bool trans_B = cur_matmul_op->get_transpose_B();

    if (matmul_a == transpose)
    {
        trans_A = !trans_A;
    }
    if (matmul_b == transpose)
    {
        trans_B = !trans_B;
    }

    auto new_matmul_op =
        std::make_shared<nnfusion::op::Dot>(cur_matmul_op->get_reduction_axes_count(),
                                            (bool)cur_matmul_op->get_reduction_axes_count(),
                                            trans_A,
                                            trans_B);
    auto new_matmul_a = matmul_a;
    auto new_matmul_b = matmul_b;
    if (matmul_a == transpose)
    {
        new_matmul_a = matmul_a->get_in_edge(0)->get_src();
    }
    if (matmul_b == transpose)
    {
        new_matmul_b = matmul_b->get_in_edge(0)->get_src();
    }
    
    auto new_matmul_gnode = graph->add_node_and_edge(
        new_matmul_op, {GNodeIndex{new_matmul_a, 0}, GNodeIndex{new_matmul_b, 0}});

    std::shared_ptr<GNode> last_node = matmul;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(new_matmul_gnode, 0, dst, y);
    }

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_transmatmul->nodes.begin(), pr_transmatmul->nodes.end());

    return RemoveNodes(nodes_to_remove, new_matmul_gnode);
}