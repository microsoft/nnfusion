#include "remove_redundant_ops.hpp"
#include "gflags/gflags.h"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(ftune_output_file);

bool RemoveRedundantOpsPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    NNFUSION_LOG(INFO) << "RemoveRedundantOpsPass Start";
    if (FLAGS_ftune_output_file == "")
        return true;
    for (auto node : graph->get_ordered_ops())
    {
        bool remove = false;
        if (node->get_op_type() == "Reshape")
        {
            auto op = std::dynamic_pointer_cast<op::Reshape>(node->get_op_ptr());
            auto out_shape = node->get_output_shape(0);
            auto in_shape = node->get_input_shape(0);
            if (in_shape == out_shape && !op->get_is_layout_change())
            {
                remove = true;
            }
        }
        else if (node->get_op_type() == "Broadcast")
        {
            auto op = std::dynamic_pointer_cast<op::Broadcast>(node->get_op_ptr());
            auto out_shape = node->get_output_shape(0);
            auto in_shape = node->get_input_shape(0);
            if (in_shape == out_shape)
            {
                remove = true;
            }
        }
        if (remove)
        {
            auto src = node->get_in_edge(0)->get_src();
            int src_id = node->get_in_edge(0)->get_src_output();
            for (auto& edge : node->get_out_edges())
            {
                if (edge->is_control_edge())
                    graph->add_control_edge(src, edge->get_dst());
                else
                    graph->add_edge(src, src_id, edge->get_dst(), edge->get_dst_input());
            }
            graph->remove_node(node);
        }
    }
    NNFUSION_LOG(INFO) << "RemoveRedundantOpsPass End";
    return true;
}