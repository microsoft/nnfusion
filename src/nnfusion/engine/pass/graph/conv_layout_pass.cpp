// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "conv_layout_pass.hpp"
#include <unordered_set>
#include "nnfusion/core/kernels/cuda_gpu/cuda_common_ops.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
// #include "nnfusion/core/operators/op_define/convolution.hpp"

DEFINE_bool(fconv_cnhw, false, "Use CNHW layout");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

void update_output(std::shared_ptr<GNode> gnode, bool to_cnhw, std::map<std::pair<std::string, size_t>, bool>& global_dict) {
    NNFUSION_LOG(INFO) << "to_cnhw: " << gnode->get_op_type() << " " << to_cnhw;
    for (int i = 0; i < gnode->get_output_size(); i++) {
        global_dict[std::make_pair(gnode->get_unique_name(), i)] = to_cnhw;
    }
    if (to_cnhw) {
        auto outputs = gnode->get_outputs();
        for (int i = 0; i < gnode->get_output_size(); i++) {
            auto dtype = outputs[i]->get_element_type();
            auto shape = outputs[i]->get_shape();
            if (shape.size() != 4) {
                NNFUSION_LOG(NNFUSION_FATAL) << "not implemented!";
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
    }
}

void update_graph(std::shared_ptr<nnfusion::graph::Graph>& graph, bool keep_output) {
    NNFUSION_LOG(INFO) << "nodes in graph";
    std::map<std::pair<std::string, size_t>, bool> is_cnhw; // The x^th output of node y is in cnhw layout
    for (auto& gnode: graph->get_ordered_ops()) {
        std::cout << gnode->get_name() << " " << gnode->get_unique_name() << " " << gnode->get_op_type() << std::endl;
        std::string op_type = gnode->get_op_type();
        bool have_4d_input_tensor = false;
        for (auto& input: gnode->get_inputs()) {
            if (input->get_shape().size() == 4)
                have_4d_input_tensor = true;
        }
        if (!have_4d_input_tensor) {
            for (size_t i = 0; i < gnode->get_output_size(); i++)
                is_cnhw[std::make_pair(gnode->get_unique_name(), i)] = false;
        }
        std::vector<bool> input_is_cnhw;
        for (auto& input_edge: gnode->get_in_edges()) {
            auto input_gnode = input_edge->get_src();
            int input_oid = input_edge->get_src_output();
            auto tensor_pair = make_pair(input_gnode->get_unique_name(), (size_t) input_oid);
            NNFUSION_CHECK(is_cnhw.find(tensor_pair) != is_cnhw.end());
            input_is_cnhw.push_back(is_cnhw[tensor_pair]);
        }
        if (nnfusion::kernels::cuda::CudaElementOpMap.find(op_type) != nnfusion::kernels::cuda::CudaElementOpMap.end()) {
            bool all_same = true;
            for (int i = 1; i < input_is_cnhw.size(); i++) {
                if (input_is_cnhw[i] != input_is_cnhw[i-1])
                    all_same = false;
            }
            if (!all_same) {
                NNFUSION_LOG(NNFUSION_FATAL) << "not implemented!";
            }
            bool to_cnhw = input_is_cnhw[0];
            update_output(gnode, to_cnhw, is_cnhw);
        } else if (op_type == "Constant" || op_type == "Parameter") {
            update_output(gnode, false, is_cnhw);
        } else if (op_type == "Convolution") {
            auto conv_op = static_pointer_cast<op::Convolution>(gnode->get_op_ptr());
            assert(input_is_cnhw.size() == 2);
            assert(gnode->get_output_size() == 1);
            if (input_is_cnhw[0]) {
                NNFUSION_LOG(INFO) << gnode << ": use CNHW conv";
                conv_op->set_data_format("CNHW");
            } else {
                NNFUSION_LOG(INFO) << gnode << ": use NCHW2CNHW conv";
                conv_op->set_data_format("NCHW2CNHW");
            }
            update_output(gnode, true, is_cnhw);
        } else if (op_type == "BatchNormInference") {
            assert(gnode->get_output_size() == 1);
            update_output(gnode, input_is_cnhw[op::BatchNormInference::INPUT_DATA], is_cnhw);
        } else {
            NNFUSION_LOG(NNFUSION_FATAL) << "layout change of op " << op_type << " is not yet supported";
        }
    }
    for (auto& gnode: graph->get_ordered_ops()) {
        std::cout << *gnode << std::endl;
    }
    if (keep_output) {
        auto indexed_outputs = graph->get_indexed_outputs();
        for (int i = 0; i < indexed_outputs.size(); i++) {
            auto& gnode_idx = indexed_outputs[i];
            if (is_cnhw[make_pair(gnode_idx.gnode->get_unique_name(), (size_t) gnode_idx.index)]) {
                // insert transpose from CNHW to NCHW
                assert(gnode_idx.gnode->get_output_shape(gnode_idx.index).size() == 4);
                auto out_gnode = nnfusion::graph::numpy_transpose(gnode_idx.gnode, {1, 0, 2, 3}, gnode_idx.index);
                std::cout << "gnode is null: " << (out_gnode == nullptr) << std::endl;
                graph->add_gnode_and_edge(out_gnode, GNodeIndexVector({gnode_idx}));
                std::cout << "XXXXX" << std::endl;
                auto out_gnode_idx = GNodeIndex(out_gnode, 0);
                graph->set_output(out_gnode_idx, i);
            }
        }
    } else {
        NNFUSION_LOG(NNFUSION_FATAL) << "not implemented!";
    }
}

bool ConvLayoutPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (!FLAGS_fconv_cnhw)
        return true;
    update_graph(graph, true);
    return true;
}
