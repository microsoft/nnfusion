
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "split_softmax_pass.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/softmax.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(ftune_output_file);

bool SplitSoftmaxPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_ftune_output_file == "")
        return true;
    for (auto node : graph->get_ordered_ops())
    {
        if (node->get_op_type() != "Softmax")
            continue;
        auto op = std::dynamic_pointer_cast<op::Softmax>(node->get_op_ptr());
        auto ax = op->get_axes();
        vector<int> ax_vec = {ax.begin(), ax.end()};
        op::OpConfig::any config[4];
        for (int i = 0; i < 4; i++)
        {
            config[i]["stage"] = i;
            config[i]["axes"] = ax_vec;
        }
        auto op0 = make_shared<op::GenericOp>(node->get_name() + ".max", "SoftmaxBasic", config[0]);
        auto op1 =
            make_shared<op::GenericOp>(node->get_name() + ".subexp", "SoftmaxBasic", config[1]);
        auto op2 = make_shared<op::GenericOp>(node->get_name() + ".sum", "SoftmaxBasic", config[2]);
        auto op3 = make_shared<op::GenericOp>(node->get_name() + ".div", "SoftmaxBasic", config[3]);
        auto input_node = node->get_in_edge(0)->get_src();
        auto node0 = graph->add_node_and_edge(op0, {input_node});
        auto node1 = graph->add_node_and_edge(op1, {input_node, node0});
        auto node2 = graph->add_node_and_edge(op2, {node1});
        auto node3 = graph->add_node_and_edge(op3, {node1, node2});
        for (auto& edge : node->get_out_edges())
        {
            if (edge->is_control_edge())
                graph->add_control_edge(node3, edge->get_dst());
            else
                graph->add_edge(node3, 0, edge->get_dst(), edge->get_dst_input());
        }
        graph->remove_node(node);
    }
    return true;
}
