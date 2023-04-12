#include "dot_algo_select_pass.hpp"
#include "gflags/gflags.h"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/core/operators/op_define/sum.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(ftune_output_file);
DECLARE_string(ffusion_skiplist);
DECLARE_bool(ftc_rewrite);

const int kSM = 80; // For V100

namespace
{
    std::unordered_set<std::string> skip_ops = {};
    void parse_skip_ops()
    {
        stringstream ss(FLAGS_ffusion_skiplist);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            skip_ops.insert(substr);
        }
    }
}

bool DotAlgoSelectPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    NNFUSION_LOG(INFO) << "DotAlgoSelectPass Start";
    parse_skip_ops();
    if (FLAGS_ftune_output_file == "" || skip_ops.count("Dot"))
        return true;
    for (auto node : graph->get_ordered_ops())
    {
        if (node->get_op_type() == "Dot")
        {
            auto op = std::dynamic_pointer_cast<op::Dot>(node->get_op_ptr());
            auto out_shape = node->get_output_shape(0);
            auto A_shape = node->get_input_shape(0);
            size_t k =
                op->get_transpose_A() ? A_shape[A_shape.size() - 2] : A_shape[A_shape.size() - 1];
            size_t num_output_elem = 1;
            for (auto n : out_shape)
                num_output_elem *= n;
            size_t split_k_factor = 4;
            if (k % split_k_factor == 0 && num_output_elem / (64 * 64) < kSM &&
                k / split_k_factor >= 32)
            {
                op::OpConfig::any config;
                config["split_k_factor"] = split_k_factor;
                config["transpose_A"] = op->get_transpose_A();
                config["transpose_B"] = op->get_transpose_B();
                config["old_out_shape"] = out_shape;
                config["tc_enabled"] =
                    FLAGS_ftc_rewrite && node->get_output_element_type(0) == nnfusion::element::f16;
                auto dot_op =
                    make_shared<op::GenericOp>(node->get_name() + ".dot", "DotSplitK", config);
                std::vector<size_t> reduction_axes;
                reduction_axes.push_back(0);
                auto reduce_op = make_shared<op::Sum>(reduction_axes);
                auto A = node->get_in_edge(0)->get_src();
                auto B = node->get_in_edge(1)->get_src();
                auto node0 = graph->add_node_and_edge(dot_op, {A, B});
                auto node1 = graph->add_node_and_edge(reduce_op, {node0});
                for (auto& edge : node->get_out_edges())
                {
                    if (edge->is_control_edge())
                        graph->add_control_edge(node1, edge->get_dst());
                    else
                        graph->add_edge(node1, 0, edge->get_dst(), edge->get_dst_input());
                }
                graph->remove_node(node);
            }
        }
    }
    NNFUSION_LOG(INFO) << "DotAlgoSelectPass End";
    return true;
}