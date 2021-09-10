#include "matmuladd_fusion_optimizer.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool MatMulAddFusionOptimizer::create_subgraphs()
{
    auto check_root = [](std::shared_ptr<GNode> gnode) -> bool {
        if (gnode->get_op_type() != "Dot")
            return false;
        auto A_shape = gnode->get_input_shape(0);
        auto B_shape = gnode->get_input_shape(1);
        if (A_shape.size() != 2 || B_shape.size() != 2)
            return false;
        if (gnode->get_out_edges().size() != 1)
            return false;
        return true;
    };

    SubGraph::Pointer s_matmuladd = std::make_shared<SubGraph>();
    s_matmuladd->name = "matmuladd";
    s_matmuladd->check_starting_node = check_root;

    {
        Pattern::Pointer p_matmuladd = std::make_shared<Pattern>();
        std::vector<std::string> ops_mul{"Dot", "Add"};
        p_matmuladd->descriptions.push_back(std::make_pair(ops_mul, 1));
        p_matmuladd->reverse_order = false;
        auto check_matmuladd = [](const PatternRecord& pr) -> bool {
            auto matmul = pr.nodes[0];
            auto add = pr.nodes[1];
            return (matmul->get_output_element_type(0) == add->get_output_element_type(0));
        };
        p_matmuladd->check.push_back(check_matmuladd);

        s_matmuladd->patterns.push_back(p_matmuladd);
    }

    m_subgraphs.push_back(s_matmuladd);
    return true;
}

bool MatMulAddFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
{
    auto pr_matmuladd = subgraph_record->pattern_records[0];
    auto matmul = pr_matmuladd->nodes[0];
    auto add = pr_matmuladd->nodes[1];
    auto matmul_a = matmul->get_in_edge(0)->get_src();
    auto matmul_b = matmul->get_in_edge(1)->get_src();

    std::shared_ptr<GNode> nodeC;
    for (auto in_edge : add->get_in_edges())
    {
        auto src = in_edge->get_src();
        if (src != matmul)
        {
            nodeC = src;
            break;
        }
    }

    // create matmuladd node
    auto matmul_op = std::dynamic_pointer_cast<op::Dot>(matmul->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(matmul_op);
    bool trans_A = matmul_op->get_transpose_A();
    bool trans_B = matmul_op->get_transpose_B();
    nnfusion::op::OpConfig::any myConfig;
    myConfig["trans_A"] = trans_A;
    myConfig["trans_B"] = trans_B;

    auto matmuladd_op = std::make_shared<nnfusion::op::GenericOp>(
        matmul->get_name() + "add", "MatMulAdd", myConfig);
    auto matmuladd_gnode = graph->add_node_and_edge(
        matmuladd_op, {GNodeIndex{matmul_a, 0}, GNodeIndex{matmul_b, 0}, GNodeIndex{nodeC, 0}});

    std::shared_ptr<GNode> last_node = add;

    auto out_edges = last_node->get_out_edges();
    for (auto out_edge : out_edges)
    {
        auto dst = out_edge->get_dst();
        int y = out_edge->get_dst_input();
        graph->remove_edge(out_edge);
        graph->add_edge(matmuladd_gnode, 0, dst, y);
    }

    std::unordered_set<std::shared_ptr<GNode>> nodes_to_remove;
    nodes_to_remove.insert(pr_matmuladd->nodes.begin(), pr_matmuladd->nodes.end());

    return RemoveNodes(nodes_to_remove, matmuladd_gnode);
}