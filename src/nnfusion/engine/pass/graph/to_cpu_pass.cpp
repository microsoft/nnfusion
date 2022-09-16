#include "to_cpu_pass.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include <queue>
#include "nnfusion/engine/engine.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_bool(fenable_cpu, false, "Run small ops on CPU");
const int cpu_thres = 64;

/*
Parameters are always regarded as large nodes.
Constants are regarded as large nodes but we can easily provide the pointer to its value on CPU.
// TODO: if a control flow node is "small", parameters in the inner graph is on CPU.
*/

const std::string stage_cpu_tag = "stage_cpu";

bool is_constant_node(std::shared_ptr<GNode> gnode) {
    if (gnode->get_op_type() == "Constant") return true;
    if (gnode->get_op_type() == "Reshape" && !std::dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr())->get_is_layout_change()) {
        bool inp_is_constant = true;
        for (auto in_edge: gnode->get_in_edges()) {
            inp_is_constant &= is_constant_node(in_edge->get_src());
            if (!inp_is_constant) return false;
        }
        return inp_is_constant;
    }
    return false;
}

std::set<std::shared_ptr<GNode>> get_small_ops(std::shared_ptr<nnfusion::graph::Graph>& graph) {
    std::set<std::shared_ptr<GNode>> small_ops;
    for (auto gnode: graph->get_nodes()) {
        bool have_large_io_node = false;
        for (int i = 0; i < gnode->get_input_size() && !have_large_io_node; i++) {
            if (shape_size(gnode->get_input_shape(i)) > cpu_thres && !is_constant_node(gnode->get_in_edge(i)->get_src())) {
                have_large_io_node = true;
                break;
            }
        }
        for (int i = 0; i < gnode->get_output_size() && !have_large_io_node; i++) {
            if (shape_size(gnode->get_output_shape(i)) > cpu_thres) {
                have_large_io_node = true;
                break;
            }
        }
        if (!have_large_io_node && gnode->get_op_type() != "Constant" && gnode->get_op_type() != "Parameter") {
            small_ops.insert(gnode);
        }
    }
    return small_ops;
}

void assign_stage(std::shared_ptr<nnfusion::graph::Graph>& graph, std::set<std::shared_ptr<GNode>> small_ops) {
    for (auto gnode: graph->get_ordered_ops()) {
        if (gnode->get_op_type() == "Constant" || gnode->get_op_type() == "Parameter") {
            gnode->Set<int>(stage_cpu_tag, 0);
            continue;
        }
        bool is_small = (small_ops.find(gnode) != small_ops.end());
        int max_large_stage = -1;
        int max_small_stage = -1;
        for (auto in_edge: gnode->get_in_edges()) {
            auto in_node = in_edge->get_src();
            if (small_ops.find(in_node) != small_ops.end()) {
                max_small_stage = max(max_small_stage, in_node->Get<int>(stage_cpu_tag));
            } else {
                max_large_stage = max(max_large_stage, in_node->Get<int>(stage_cpu_tag));
            }
        }
        int my_stage;
        if (is_small) {
            if (max_small_stage < max_large_stage) {
                my_stage = max_large_stage + 1;
            } else {
                my_stage = max(max_small_stage, 1);
            }
        } else {
            if (max_large_stage < max_small_stage) {
                my_stage = max_small_stage + 1;
            } else {
                my_stage = max(max_large_stage, 0);
            }
        }
        NNFUSION_CHECK(my_stage % 2 == is_small);
        gnode->Set<int>(stage_cpu_tag, (int) my_stage);
    }
    for (auto gnode: graph->get_nodes()) {
        if (!gnode->hasAttribute(stage_cpu_tag)) {
            gnode->Set<int>(stage_cpu_tag, (int) 0);
        }
    }
}

void const_propogate(std::shared_ptr<nnfusion::graph::Graph>& graph, std::set<std::shared_ptr<GNode>> small_ops) {
    auto ordered_ops = graph->get_ordered_ops();
    for (auto it = ordered_ops.rbegin(); it != ordered_ops.rend(); ++it) {
        auto gnode = *it;
        if (is_constant_node(gnode)) {
            auto min_stage = INT_MAX;
            for (auto out_edge: gnode->get_out_edges()) {
                auto out_gnode = out_edge->get_dst();
                min_stage = min(min_stage, out_gnode->Get<int>(stage_cpu_tag));
            }
            gnode->Set<int>(stage_cpu_tag, (int) min_stage);
        }
    }
    for (auto gnode: ordered_ops) {
        if (gnode->get_op_type() == "Result") {
            auto max_stage = 0;
            for (auto in_edge: gnode->get_in_edges()) {
                auto in_gnode = in_edge->get_src();
                max_stage = max(max_stage, in_gnode->Get<int>(stage_cpu_tag));
            }
            gnode->Set<int>(stage_cpu_tag, (int) ceil_div(max_stage, 2) * 2);
        }
    }
}

void add_copy_node(std::shared_ptr<nnfusion::graph::Graph>& graph) {
    for (auto gnode: graph->get_ordered_ops()) {
        for (auto out_edge: gnode->get_out_edges()) {
            auto out_node = out_edge->get_dst();
            bool src_on_cpu = gnode->Get<int>(stage_cpu_tag) & 1;
            bool dst_on_cpu = out_node->Get<int>(stage_cpu_tag) & 1;
            if (src_on_cpu != dst_on_cpu) {
                std::shared_ptr<op::Op> op = src_on_cpu ? dynamic_pointer_cast<op::Op>(std::make_shared<op::H2D>()) : dynamic_pointer_cast<op::Op>(std::make_shared<op::D2H>());
                auto copy_node = graph->add_node_and_edge(op, {GNodeIndex(gnode, out_edge->get_src_output())}, 1);
                copy_node->get_op_ptr()->revalidate_and_infer_types(copy_node);
                copy_node->Set<int>(stage_cpu_tag, (int) gnode->Get<int>(stage_cpu_tag));
                graph->add_edge(copy_node, 0, out_node, out_edge->get_dst_input());
                graph->remove_edge(out_edge);
            }
        }
    }
}

bool ToCPUPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) {
    auto ctx = get_context();
    bool is_outmost_graph = ctx != nullptr && ctx->is_outmost_graph;
    // to_cpu is not fully supported in inner graph yet
    if (!FLAGS_fenable_cpu || !is_outmost_graph) {
        for (auto gnode: graph->get_nodes()) {
            gnode->set_on_gpu(true);
            gnode->Set<int>(stage_cpu_tag, 0);
        }
        return true;
    }
    auto small_ops = get_small_ops(graph);
    assign_stage(graph, small_ops);
    const_propogate(graph, small_ops);
    add_copy_node(graph);
    for (auto gnode: graph->get_ordered_ops()) {
        if (gnode->Get<int>(stage_cpu_tag) & 1) {
            gnode->set_on_gpu(false);
        } else {
            gnode->set_on_gpu(true);
        }
    }
    return true;
}
