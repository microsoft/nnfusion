// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "conv_layout_pass.hpp"
#include <unordered_set>
#include "nnfusion/core/kernels/cuda_gpu/cuda_common_ops.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
// #include "nnfusion/core/operators/op_define/convolution.hpp"

DEFINE_bool(fconv_cnhw, false, "Use CNHW layout");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

void update_output(std::shared_ptr<GNode> gnode, std::vector<bool> to_cnhw, std::map<std::pair<std::string, size_t>, bool>& global_dict) {
    NNFUSION_LOG(INFO) << "to_cnhw: " << *gnode << " " << gnode->get_unique_name() << " " << to_cnhw[0];
    for (int i = 0; i < gnode->get_output_size(); i++) {
        global_dict[std::make_pair(gnode->get_unique_name(), i)] = to_cnhw[i];
    }
    auto outputs = gnode->get_outputs();
    std::vector<Shape> output_shapes;
    for (int i = 0; i < gnode->get_output_size(); i++) {
        if (to_cnhw[i]) {
            auto dtype = outputs[i]->get_element_type();
            auto shape = outputs[i]->get_shape();
            if (shape.size() != 4) {
                NNFUSION_CHECK_FAIL() << "not implemented!";
            }
            std::swap(shape[0], shape[1]);
            gnode->set_output_type_and_shape(i, dtype, shape);
            auto users = gnode->get_output_users(i);
            for (auto& user_edge: users) {
                auto dst_gnode = user_edge->get_dst();
                auto dst_id = user_edge->get_dst_input();
                NNFUSION_LOG(INFO) << "change input shape " << dst_gnode->get_name() << " " << dst_id << " from " << dst_gnode->get_input_shape(dst_id) << " to "  << shape;
                dst_gnode->set_input(dst_id, std::make_shared<Input>(dtype, PartialShape(shape)));
            }
        }
        output_shapes.push_back(gnode->get_output_shape(i));
    }

    if (gnode->get_op_type() != "Convolution" && gnode->get_op_type() != "BatchNormInference") { // cannot pass the validation of these ops, skip it to avoid validation fail
        gnode->get_op_ptr()->revalidate_and_infer_types(gnode);
        gnode->get_op_ptr()->infer_shared_memory(gnode);
        for (int i = 0; i < gnode->get_output_size(); i++) {
            NNFUSION_CHECK(gnode->get_output_shape(i) == output_shapes[i]) << "output " << i << " " << output_shapes[i] << " " << gnode->get_output_shape(i);
        }
    }
    NNFUSION_LOG(INFO) << "after to_cnhw: " << *gnode << " " << gnode->get_unique_name() << " " << to_cnhw[0];
}

void update_output(std::shared_ptr<GNode> gnode, bool to_cnhw, std::map<std::pair<std::string, size_t>, bool>& global_dict) {
    NNFUSION_CHECK(gnode->get_output_size() == 1);
    update_output(gnode, std::vector<bool>{to_cnhw}, global_dict);
}

void transpose_const_op(std::shared_ptr<op::Constant> op) {
    NNFUSION_LOG(NNFUSION_WARNING) << "skip data transpose ADHOC for concat op of resnet";
    auto shape = op->get_shape();
    swap(shape[0], shape[1]);
    op->set_shape(shape);
}

// is_cnhw: The x^th output of node y is in cnhw layout
void update_graph(std::shared_ptr<nnfusion::graph::Graph>& graph, std::map<std::pair<std::string, size_t>, bool>& is_cnhw, bool out_most_graph) {
    NNFUSION_LOG(INFO) << "nodes in graph";
    for (auto& gnode: graph->get_ordered_ops()) {
        std::cout << *gnode << std::endl;
    }
    
    for (auto& gnode: graph->get_ordered_ops()) {
        std::cout << gnode->get_name() << " " << gnode->get_unique_name() << " " << gnode->get_op_type() << std::endl;
        std::string op_type = gnode->get_op_type();
        if (op_type == "Parameter" && !out_most_graph) {
            NNFUSION_CHECK(is_cnhw.find(make_pair(gnode->get_unique_name(), 0)) != is_cnhw.end());
            if (is_cnhw[std::make_pair(gnode->get_unique_name(), 0)]) {
                auto param_op = dynamic_pointer_cast<op::Parameter>(gnode->get_op_ptr());
                auto shape = param_op->get_shape();
                swap(shape[0], shape[1]);
                param_op->set_shape(shape);
            }
            update_output(gnode, is_cnhw[std::make_pair(gnode->get_unique_name(), 0)], is_cnhw);
            continue;
        }
    
        bool have_4d_input_tensor = false;
        for (auto& input: gnode->get_inputs()) {
            if (input->get_shape().size() == 4)
                have_4d_input_tensor = true;
        }
        if (!have_4d_input_tensor) {
            for (size_t i = 0; i < gnode->get_output_size(); i++)
                is_cnhw[std::make_pair(gnode->get_unique_name(), i)] = false;
            continue;
        }
        std::vector<bool> input_is_cnhw;
        for (auto& input_edge: gnode->get_in_edges()) {
            auto input_gnode = input_edge->get_src();
            int input_oid = input_edge->get_src_output();
            auto tensor_pair = make_pair(input_gnode->get_unique_name(), (size_t) input_oid);
            NNFUSION_CHECK(is_cnhw.find(tensor_pair) != is_cnhw.end());
            input_is_cnhw.push_back(is_cnhw[tensor_pair]);
        }
        std::set<std::string> keep_data_format_ops = {"AvgPool", "MaxPool"};
        if (nnfusion::kernels::cuda::CudaElementOpMap.find(op_type) != nnfusion::kernels::cuda::CudaElementOpMap.end() ||
            keep_data_format_ops.find(op_type) != keep_data_format_ops.end()) {
            bool all_same = true;
            for (int i = 1; i < input_is_cnhw.size(); i++) {
                if (input_is_cnhw[i] != input_is_cnhw[i-1])
                    all_same = false;
            }
            int op_use_cnhw = input_is_cnhw[0];
            if (!all_same) {
                for (int i = 0; i < input_is_cnhw.size(); i++) {
                    NNFUSION_CHECK(gnode->get_input_shape(i).size() == 4) << "only consider 4d tensor now";
                    std::string input_op_type = gnode->get_in_edge(i)->get_src()->get_op_type();
                    if (input_op_type != "Constant" && input_op_type != "BatchNormInference" && input_op_type != "Broadcast") {
                        if (op_use_cnhw == -1) {
                            op_use_cnhw = input_is_cnhw[i];
                        } else {
                            // use cnhw if any input is cnhw
                            op_use_cnhw |= input_is_cnhw[i];
                        }
                    }
                }
                NNFUSION_LOG(INFO) << "use_cnhw " << *gnode << op_use_cnhw;
                if (op_use_cnhw == -1) op_use_cnhw = 0; // assign nchw layout if all input layouts are undefined
                if (op_use_cnhw) {
                    for (int i = 0; i < input_is_cnhw.size(); i++) {
                        if (input_is_cnhw[i] == op_use_cnhw) continue;
                        if (gnode->get_in_edge(i)->get_src()->is_constant()) {
                            auto const_node = gnode->get_in_edge(i)->get_src();
                            auto const_op = std::dynamic_pointer_cast<op::Constant>(const_node->get_op_ptr());
                            transpose_const_op(const_op);
                            update_output(const_node, true, is_cnhw);
                        } else if (gnode->get_in_edge(i)->get_src()->get_op_type() == "BatchNormInference") {
                            auto bn_node = gnode->get_in_edge(i)->get_src();
                            update_output(bn_node, true, is_cnhw);
                        } else if (gnode->get_in_edge(i)->get_src()->get_op_type() == "Broadcast") {
                            auto bcast_node = gnode->get_in_edge(i)->get_src();
                            auto bcast_op = std::dynamic_pointer_cast<op::Broadcast>(bcast_node->get_op_ptr());
                            Shape shape = bcast_op->get_broadcast_shape();
                            NNFUSION_CHECK(shape.size() == 4);
                            swap(shape[0], shape[1]);
                            AxisSet axis = bcast_op->get_broadcast_axes();
                            bool bcast_d0 = (axis.find(0) != axis.end());
                            bool bcast_d1 = (axis.find(1) != axis.end());
                            axis.erase(0);
                            axis.erase(1);
                            if (bcast_d0) { axis.insert(1); }
                            if (bcast_d1) { axis.insert(0); }
                            bcast_op->set_broadcast_axes(axis);
                            bcast_op->set_broadcast_shape(shape);
                            NNFUSION_LOG(INFO) << "axisset: " << bcast_op->get_broadcast_axes() << " shape: " << bcast_op->get_broadcast_shape() << " node: " << *bcast_node;
                            update_output(bcast_node, true, is_cnhw);
                            NNFUSION_LOG(INFO) << "bcast inner outer: " << bcast_op->is_inner_broadcast() << " " << bcast_op->is_outer_broadcast();
                        }
                    }
                }
            }
            update_output(gnode, op_use_cnhw, is_cnhw);
        } else if (op_type == "Constant") {
            update_output(gnode, false, is_cnhw);
        } else if (op_type == "Parameter") {
            update_output(gnode, false, is_cnhw);
        } else if (op_type == "Convolution") {
            auto conv_op = static_pointer_cast<op::Convolution>(gnode->get_op_ptr());
            assert(input_is_cnhw.size() == 2);
            assert(gnode->get_output_size() == 1);
            if (input_is_cnhw[0]) {
                NNFUSION_LOG(INFO) << *gnode << ": use CNHW conv";
                conv_op->set_data_format("CNHW");
            } else {
                NNFUSION_LOG(INFO) << *gnode << ": use NCHW2CNHW conv";
                conv_op->set_data_format("NCHW2CNHW");
            }
            update_output(gnode, true, is_cnhw);
        } else if (op_type == "BatchNormInference") {
            assert(gnode->get_output_size() == 1);
            update_output(gnode, input_is_cnhw[op::BatchNormInference::INPUT_DATA], is_cnhw);
        } else if (op_type == "Concat") {
            int op_use_cnhw = -1;
            for (int i = 0; i < input_is_cnhw.size(); i++) {
                NNFUSION_CHECK(gnode->get_input_shape(i).size() == 4) << "only consider 4d tensor now";
                if (!gnode->get_in_edge(i)->get_src()->is_constant()) {
                    if (op_use_cnhw == -1) {
                        op_use_cnhw = input_is_cnhw[i];
                    } else if (op_use_cnhw != input_is_cnhw[i]) {
                        NNFUSION_CHECK_FAIL() << "not implemented";
                    }
                }
            }
            if (op_use_cnhw == -1) op_use_cnhw = 0; // assign nchw layout if all inputs are constant
            if (op_use_cnhw) {
                for (int i = 0; i < input_is_cnhw.size(); i++) {
                    if (gnode->get_in_edge(i)->get_src()->is_constant()) {
                        auto const_node = gnode->get_in_edge(i)->get_src();
                        auto const_op = std::dynamic_pointer_cast<op::Constant>(const_node->get_op_ptr());
                        transpose_const_op(const_op);
                        update_output(const_node, true, is_cnhw);
                    }
                }
                auto concat_op = std::dynamic_pointer_cast<op::Concat>(gnode->get_op_ptr());
                size_t concat_axis = concat_op->get_concatenation_axis();
                if (concat_axis == 0 || concat_axis == 1) {
                    NNFUSION_LOG(INFO) << *gnode << " change concat axis from " << concat_axis << " to " << 1 - concat_axis;
                    concat_op->set_concatenation_axis(1 - concat_axis);
                }
            }
            update_output(gnode, op_use_cnhw, is_cnhw);
        } else if (op_type == "Reshape") {
            auto op = std::dynamic_pointer_cast<op::Reshape>(gnode->get_op_ptr());
            if (gnode->get_output_shape(0).size() == 4) {
                NNFUSION_CHECK_FAIL() << "not implemented";
                // TODO: keep cnhw when reshape to a 4d tensor. Be careful of out_shape attribute of reshape_op
            } else {
                AxisVector order = op->get_input_order();
                for (auto& o: order)
                    if (o == 0 || o == 1) { o = 1 - o; }
                Shape shape = op->get_output_shape();
                NNFUSION_LOG(INFO) << *gnode << " change order to " << order;
                auto new_reshape_op = std::make_shared<op::Reshape>(order, shape);
                new_reshape_op->set_name(op->get_name());
                gnode->construct_from_op_ptr(new_reshape_op);
                update_output(gnode, false, is_cnhw );
            }
        } else if (op_type == "Sum") {
            auto op = std::dynamic_pointer_cast<op::Sum>(gnode->get_op_ptr());
            AxisSet axis = op->get_reduction_axes();
            bool reduce_d0 = (axis.find(0) != axis.end());
            bool reduce_d1 = (axis.find(1) != axis.end());
            if (reduce_d0 != reduce_d1) NNFUSION_CHECK_FAIL() << "not implemented";
            update_output(gnode, input_is_cnhw[0], is_cnhw);
        } else if (op_type == "If") {
            auto op = std::dynamic_pointer_cast<op::If>(gnode->get_op_ptr());
            auto is_cnhw_then_branch = is_cnhw;
            auto then_graph = op->get_then_branch_graph();
            NNFUSION_LOG(INFO) << "nodes in then graph";
            for (auto& subgraph_gnode: then_graph->get_ordered_ops()) {
                std::cout << *subgraph_gnode << std::endl;
                if (std::dynamic_pointer_cast<op::Parameter>(subgraph_gnode->get_op_ptr()) != nullptr) {
                    auto in_edge = gnode->get_in_edge(subgraph_gnode->Get<int>("subgraph_input_map"));
                    bool param_is_cnhw = is_cnhw_then_branch[std::make_pair(in_edge->get_src()->get_unique_name(), (size_t) in_edge->get_src_output())];
                    std::cout << "subgraph_param: " << *subgraph_gnode << " " << "unique name: " << subgraph_gnode->get_unique_name() << " is cnhw: " << param_is_cnhw;
                    is_cnhw_then_branch[std::make_pair(subgraph_gnode->get_unique_name(), 0)] = param_is_cnhw;
                }
            }
            update_graph(then_graph, is_cnhw_then_branch, false);

            auto is_cnhw_else_branch = is_cnhw;
            auto else_graph = op->get_else_branch_graph();
            NNFUSION_LOG(INFO) << "nodes in else graph";
            for (auto& subgraph_gnode: else_graph->get_ordered_ops()) {
                std::cout << *subgraph_gnode << std::endl;
                if (std::dynamic_pointer_cast<op::Parameter>(subgraph_gnode->get_op_ptr()) != nullptr) {
                    auto in_edge = gnode->get_in_edge(subgraph_gnode->Get<int>("subgraph_input_map"));
                    bool param_is_cnhw = is_cnhw_else_branch[std::make_pair(in_edge->get_src()->get_unique_name(), (size_t) in_edge->get_src_output())];
                    is_cnhw_else_branch[std::make_pair(subgraph_gnode->get_unique_name(), 0)] = param_is_cnhw;
                }
            }
            update_graph(else_graph, is_cnhw_else_branch, false);
            // TODO: merge two is_cnhw
            auto then_output = then_graph->get_indexed_outputs();
            auto else_output = else_graph->get_indexed_outputs();
            std::vector<bool> output_is_cnhw;
            for (int i = 0; i < then_output.size(); i++) {
                bool then_cnhw = is_cnhw_then_branch[make_pair(then_output[i].gnode->get_unique_name(), (size_t) then_output[i].index)];
                bool else_cnhw = is_cnhw_else_branch[make_pair(else_output[i].gnode->get_unique_name(), (size_t) else_output[i].index)];
                NNFUSION_CHECK(then_cnhw == else_cnhw) << "not implemented";
                output_is_cnhw.push_back(then_cnhw);
                if (then_cnhw) {
                    auto out_shape = op->get_output_shape(i);
                    swap(out_shape[0], out_shape[1]);
                    op->set_output_shape(i, out_shape);
                }
            }
            NNFUSION_LOG(INFO) << "if op: output_is_cnhw";
            for (auto x: output_is_cnhw) { std::cout << x << " "; } std::cout << std::endl;
            update_output(gnode, output_is_cnhw, is_cnhw);
        } else {
            NNFUSION_CHECK_FAIL() << "layout change of op " << *gnode << " is not yet supported";
        }
    }
    if (out_most_graph) {
        auto indexed_outputs = graph->get_indexed_outputs();
        for (int i = 0; i < indexed_outputs.size(); i++) {
            auto& gnode_idx = indexed_outputs[i];
            if (is_cnhw[make_pair(gnode_idx.gnode->get_unique_name(), (size_t) gnode_idx.index)]) {
                // insert transpose from CNHW to NCHW
                assert(gnode_idx.gnode->get_output_shape(gnode_idx.index).size() == 4);
                auto out_gnode = nnfusion::graph::numpy_transpose(gnode_idx.gnode, {1, 0, 2, 3}, gnode_idx.index);
                graph->add_gnode_and_edge(out_gnode, GNodeIndexVector({gnode_idx}));
                auto out_gnode_idx = GNodeIndex(out_gnode, 0);
                graph->set_output(out_gnode_idx, i);
            }
        }
    }
    NNFUSION_LOG(INFO) << "after conv layout pass:";
    for (auto& gnode: graph->get_ordered_ops()) {
        std::cout << *gnode << std::endl;
    }
}

bool ConvLayoutPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fconv_cnhw)
        return true;
    std::map<std::pair<std::string, size_t>, bool> is_cnhw;
    update_graph(graph, is_cnhw, true);
    return true;
}
