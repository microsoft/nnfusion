#include "subgraph_op_move.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/core/operators/op_define/if_single.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_bool(fbranch_split, false, "Move small ops of if branch out");

const int move_thres = 128;

void SubGraphOpMovePass::find_and_move_small_op_out(
    std::shared_ptr<nnfusion::graph::Graph> full_graph,
    std::shared_ptr<nnfusion::graph::Graph> sub_graph,
    std::shared_ptr<nnfusion::graph::GNode> if_node,
    bool is_then_branch) {
    std::map<std::string, bool> moved;
    auto gnodes = sub_graph->get_ordered_ops();
    GNodeVector to_move;
    // std::vector<std::shared_ptr<nnfusion::graph::GNode>> to_move;

    for (auto gnode: gnodes) {
        bool should_move = true;
        if (gnode->get_op_type() == "Parameter" || gnode->get_op_type() == "Constant") continue;
        for (auto in_edge: gnode->get_in_edges()) {
            auto in_gnode = in_edge->get_src();
            if (in_gnode->get_op_type() != "Parameter" && in_gnode->get_op_type() != "Constant" && !moved[in_gnode->get_unique_name()]) {
                should_move = false;
            }
        }
        for (auto input: gnode->get_inputs()) {
            if (shape_size(input->get_shape()) > move_thres) {
                should_move = false;
            }
        }
        for (auto output: gnode->get_outputs()) {
            if (shape_size(output->get_shape()) > move_thres) {
                should_move = false;
            }
        }
        if (should_move) {
            to_move.push_back(gnode);
            moved[gnode->get_unique_name()] = true;
            std::cout << *gnode << std::endl;
        }
    }

    NNFUSION_LOG(INFO) << "to_move gnodes";
    for (auto gnode: to_move) {
        std::cout << *gnode << std::endl;
    }
    std::cout << "---------------------\n";

    move_out(to_move, if_node, full_graph, sub_graph, is_then_branch);
}

void SubGraphOpMovePass::move_out(GNodeVector to_move, std::shared_ptr<GNode> if_node, std::shared_ptr<Graph> full_graph, std::shared_ptr<Graph> sub_graph, bool is_then_branch) {
    std::map<std::string, std::shared_ptr<GNode>> moved_gnodes;
    // create gnode in full_graph for each gnode in to_move
    for (auto gnode: to_move) {
        GNodeVector new_inputs;
        std::shared_ptr<op::Op> new_op;
        if (gnode->get_op_type() == "Reshape" && !dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr())->get_is_layout_change()) {
            new_op = gnode->get_op_ptr();
        } else {
            new_op = std::make_shared<op::IfSingle>(gnode->get_op_ptr(), is_then_branch);
            new_inputs.push_back(if_node->get_in_edge(0)->get_src());
        }
        for (auto in_edge: gnode->get_in_edges()) {
            auto in_gnode = in_edge->get_src();
            if (in_gnode->get_op_type() == "Parameter") {
                new_inputs.push_back(if_node->get_in_edge(in_gnode->Get<int>("subgraph_input_map"))->get_src());
            } else if (moved_gnodes.find(in_gnode->get_unique_name()) != moved_gnodes.end()) {
                new_inputs.push_back(moved_gnodes[in_gnode->get_unique_name()]);
            } else if (in_gnode->get_op_type() == "Constant") {
                auto inner_op = dynamic_pointer_cast<op::Constant>(in_gnode->get_op_ptr());
                auto const_op = std::make_shared<op::Constant>(inner_op->get_type(), inner_op->get_shape(), inner_op->get_data_ptr());
                auto const_node = full_graph->add_node_and_edge(const_op, GNodeVector());
                moved_gnodes[in_gnode->get_unique_name()] = const_node;
                new_inputs.push_back(const_node);
            } else {
                NNFUSION_CHECK_FAIL() << "unreachable!";
            }
        }
        auto new_node = full_graph->add_node_and_edge(new_op, new_inputs); // op is used by two gnodes now, but the one in inner graph will be deleted later
        moved_gnodes[gnode->get_unique_name()] = new_node;
    }

    std::map<std::string, std::shared_ptr<GNode>> new_params;

    // change edge of (moved)->(not moved) in subgraph to (parameter)->(not moved)
    for (auto gnode: sub_graph->get_ordered_ops()) {
        if (moved_gnodes.find(gnode->get_unique_name()) != moved_gnodes.end()) continue;
        for (auto in_edge: gnode->get_in_edges()) {
            auto in_gnode = in_edge->get_src();
            auto in_unique_name = in_gnode->get_unique_name();
            if (moved_gnodes.find(in_unique_name) != moved_gnodes.end()) {
                if (new_params.find(in_unique_name) == new_params.end()) {
                    // add a new input to if_op
                    // process subgraph
                    auto in_tensor = in_gnode->get_output_tensor_ptr(in_edge->get_src_output());
                    auto param_op = std::make_shared<op::Parameter>(in_tensor->get_element_type(), in_tensor->get_shape());
                    auto param_gnode = sub_graph->add_node_and_edge(param_op, GNodeVector());
                    new_params[in_unique_name] = param_gnode;
                    int param_id = if_node->get_input_size();
                    param_gnode->Set<int>("subgraph_input_map", int(param_id));
                    // process fullgraph
                    if_node->set_input(param_id, std::make_shared<Input>(in_tensor->get_element_type(), in_tensor->get_partial_shape()));
                    full_graph->add_edge(moved_gnodes[in_unique_name], in_edge->get_src_output(), if_node, param_id);

                    new_params[in_unique_name] = param_gnode;
                }
                gnode->remove_in_edge(in_edge);
                sub_graph->add_edge(new_params[in_unique_name], 0, gnode, in_edge->get_dst_input());
            }
        }
    }

    // remove the moved nodes from subgraph
    for (auto gnode: to_move) {
        sub_graph->remove_node(gnode);
    }
}

bool SubGraphOpMovePass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fbranch_split) return true;
    for (auto& gnode: graph->get_ordered_ops()) {
        if (gnode->get_op_type() == "If") {
            auto if_op = dynamic_pointer_cast<op::If>(gnode->get_op_ptr());
            find_and_move_small_op_out(graph, if_op->get_then_branch_graph(), gnode, true);
            find_and_move_small_op_out(graph, if_op->get_else_branch_graph(), gnode, false);
        }
    }
    return true;
}
