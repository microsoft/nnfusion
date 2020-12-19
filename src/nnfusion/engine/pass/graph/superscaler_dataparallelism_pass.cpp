// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "superscaler_dataparallelism_pass.hpp"
#include <unistd.h>
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/allreduce.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"
using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;
using namespace std;
DEFINE_bool(fadd_sc_allreduce, false, "Add Allreduce operater after ApplyGradient operator.");
DEFINE_bool(fadd_sc_allreduce_fusion,
            false,
            "Add fused sc Allreduce operater after ApplyGradient operator.");
DEFINE_int32(sc_allreduce_fusion_num, -1, "set the number of adjacent allreduce_op to fuse.");
DEFINE_int32(sc_allreduce_fusion_size,
             -1,
             "set the floats of data to fuse: 67108864 is recommended.");
DEFINE_int32(sc_allreduce_fusion_time, -1, "set the timeout to fuse: 1000 millisecond by default.");

#define SC_ALLREDUCE_DEBUG
int SuperScalerDataParallelismPass::get_gradient_from_apply(std::shared_ptr<GNode> apply_node)
{
    // TODO(gbxu): adapt for more apply op. A general way: provide API of quering "grad" from op def.
    if (apply_node->get_op_type() == "ApplyGradientDescent" ||
        apply_node->get_op_type() == "ApplyGradient")
    {
        int weight_index = (apply_node->get_in_edge(0)->get_src()->is_variable() ||
                            (apply_node->get_in_edge(0)->get_src()->is_parameter() &&
                             std::dynamic_pointer_cast<op::Parameter>(
                                 apply_node->get_in_edge(0)->get_src()->get_op_ptr())
                                 ->require_grad()))
                               ? 0
                               : 1;
        return (weight_index + 1) % 2;
    }
    else
    {
        return -1;
    }
}

std::vector<std::vector<int>> SuperScalerDataParallelismPass::group_gradient_apply(
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    std::vector<std::vector<int>> gradient_key_subgroups;
    int sc_allreduce_fusion_num = FLAGS_sc_allreduce_fusion_num;
    int sc_allreduce_fusion_size = FLAGS_sc_allreduce_fusion_size;
    int sc_allreduce_fusion_time = FLAGS_sc_allreduce_fusion_time;
    NNFUSION_LOG(INFO) << "[sc dp pass] sc_allreduce_fusion_num:" << sc_allreduce_fusion_num;
    NNFUSION_LOG(INFO) << "[sc dp pass] sc_allreduce_fusion_size:" << sc_allreduce_fusion_size;
    NNFUSION_LOG(INFO) << "[sc dp pass] sc_allreduce_fusion_time:" << sc_allreduce_fusion_time;
    if (sc_allreduce_fusion_num <= 0 && sc_allreduce_fusion_size <= 0 &&
        sc_allreduce_fusion_time <= 0)
    {
        sc_allreduce_fusion_num = hash_to_gradient_apply.size(); // concat all allreduce into one
        NNFUSION_LOG(INFO) << "[sc dp pass] reset sc_allreduce_fusion_num:"
                           << sc_allreduce_fusion_num;
    }
    std::vector<int> subgroup;
    std::vector<int> fused_sizes;
    if (sc_allreduce_fusion_num > 0)
    {
        int curr_fuse_size = 0;
        for (int i = 0; i < hash_to_gradient_apply.size(); i++)
        {
            int curr = shape_size(hash_to_gradient_apply[i].first->get_shape());
            curr_fuse_size += curr;
            // allreduce nodes are adjacent and sorted from back to front when backward by default
            subgroup.push_back(i);
            if (subgroup.size() >= sc_allreduce_fusion_num)
            {
                gradient_key_subgroups.push_back(subgroup);
                fused_sizes.push_back(curr_fuse_size);
                subgroup.clear();
                curr_fuse_size = 0;
            }
        }
        if (subgroup.size() != 0) // fuse the remaining allreduce nodes
        {
            gradient_key_subgroups.push_back(subgroup);
            fused_sizes.push_back(curr_fuse_size);
            subgroup.clear();
            curr_fuse_size = 0;
        }
    }
    else
    {
        // timeout and buffer_size
        NNFUSION_CHECK(sc_allreduce_fusion_time == -1)
            << "now sc_allreduce_fusion_time is not supported.";
        int curr_fuse_size = 0;
        for (int i = 0; i < hash_to_gradient_apply.size(); i++)
        {
            // TODO: timeout mechanism
            int curr = shape_size(hash_to_gradient_apply[i].first->get_shape());
            if (curr_fuse_size + curr > sc_allreduce_fusion_size)
            {
                gradient_key_subgroups.push_back(subgroup);
                fused_sizes.push_back(curr_fuse_size);
                subgroup.clear();
                curr_fuse_size = 0;
            }
            subgroup.push_back(i);
            curr_fuse_size += curr;
        }
        if (subgroup.size() != 0) // fuse the remaining allreduce nodes
        {
            gradient_key_subgroups.push_back(subgroup);
            fused_sizes.push_back(curr_fuse_size);
            subgroup.clear();
            curr_fuse_size = 0;
        }
    }
    return gradient_key_subgroups;
}

std::shared_ptr<GNode> SuperScalerDataParallelismPass::concat_into_one(
    std::shared_ptr<Graph>& graph,
    std::vector<int> subgroup,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    // gradient->reshape->concat
    GNodeIndexVector concat_inputs;
    for (int i : subgroup)
    {
        auto gradient_apply = hash_to_gradient_apply[i];
        auto gradient_node = gradient_apply.first;
        int n = 0;
        nnfusion::AxisVector order = nnfusion::AxisVector(gradient_node->get_shape().size());
        std::generate(order.begin(), order.end(), [&n]() { return n++; });
        auto apply_node = gradient_apply.second;
        auto reshape_op = std::make_shared<op::Reshape>(
            order,
            nnfusion::Shape(1, shape_size(gradient_node->get_shape()))); // AxisVector={0, 1..}
        add_inplace(reshape_op, 0, 0, false);
        auto reshape_node = graph->add_node_and_edge(reshape_op, {gradient_node}); // output_index=0
        concat_inputs.push_back(GNodeIndex(reshape_node, 0));
    }
    auto concat_op = std::make_shared<nnfusion::op::Concat>(0);
    auto first_gradient_node = hash_to_gradient_apply[subgroup[0]].first;
    concat_op->set_name(first_gradient_node->get_name() + "_fusion_concat_node");
    std::shared_ptr<GNode> concat_node = graph->add_node_and_edge(concat_op, {concat_inputs});
    return concat_node;
}

std::vector<std::pair<std::shared_ptr<GNode>, int>> SuperScalerDataParallelismPass::split_from_one(
    std::shared_ptr<Graph>& graph,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply,
    std::shared_ptr<GNode> allreduce_node,
    std::vector<int> subgroup)
{
    std::vector<std::pair<std::shared_ptr<GNode>, int>> allreduced_gradients_index;
    size_t cursor = 0;
    std::vector<size_t> lower{0};
    std::vector<size_t> upper{0};
    size_t allreduced_tensor_size = shape_size(allreduce_node->get_shape());
    for (int i : subgroup)
    {
        auto gradient_apply = hash_to_gradient_apply[i];
        auto gradient_node = gradient_apply.first;
        // allreduce->slice
        nnfusion::Shape gradient_shape =
            gradient_node->get_shape(); // default get_output_shape(output_index=0)
        cursor += shape_size(gradient_shape);
        upper[0] = cursor;
        NNFUSION_CHECK(cursor <= allreduced_tensor_size) << "slice range is out of buffer";
        auto slice_op = std::make_shared<nnfusion::op::Slice>(lower, upper);
        lower[0] = cursor;
        slice_op->set_name(gradient_node->get_name() + "_fusion_slice_node");
        auto slice_node = graph->add_node_and_edge(slice_op, {allreduce_node});
        // allreduce->slice->reshape
        auto reshape_op = std::make_shared<op::Reshape>(nnfusion::AxisVector{0},
                                                        gradient_shape); // AxisVector={0, 1..}
        add_inplace(reshape_op, 0, 0, false);
        auto reshape_node = graph->add_node_and_edge(reshape_op, {slice_node}); // output_index=0
        allreduced_gradients_index.push_back(
            std::pair<std::shared_ptr<GNode>, int>(reshape_node, i));
    }
    return allreduced_gradients_index;
}

bool SuperScalerDataParallelismPass::add_allreduce(
    std::shared_ptr<Graph>& graph,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    for (int i = 0; i < hash_to_gradient_apply.size(); i++)
    {
        auto gradient_node = hash_to_gradient_apply[i].first;
        auto apply_node = hash_to_gradient_apply[i].second;
        int gradient_index = get_gradient_from_apply(apply_node);
        graph->remove_edge(apply_node->get_in_edge(gradient_index));
        // Weight(weight_node) ----|
        //                         |
        //                         V
        // (gradient)    ApplyGradient-> Result
        auto allreduce_op = std::make_shared<AllReduce>();
        std::shared_ptr<GNode> allreduce_node =
            graph->add_node_and_edge(allreduce_op, {gradient_node});
        NNFUSION_LOG(INFO) << "[sc dp pass] allreduce name:" << allreduce_node->get_name();
        // Weight(weight_node) ------------|
        //                                 |
        //                                 V
        // (gradient) --> allreduce    ApplyGradient-> Result
        graph->add_edge(allreduce_node, 0, apply_node, gradient_index);
        // Weight(weight_node) ------------|
        //                                 |
        //                                 V
        // (gradient) --> allreduce --> ApplyGradient-> Result
    }
}

bool SuperScalerDataParallelismPass::add_fused_allreduce(
    std::shared_ptr<Graph>& graph,
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply)
{
    std::vector<std::vector<int>> gradient_key_subgroups =
        group_gradient_apply(hash_to_gradient_apply);
    for (std::vector<int> subgroup : gradient_key_subgroups)
    {
        auto concat_node = concat_into_one(graph, subgroup, hash_to_gradient_apply);
        // Weight(weight_node) ----------------|
        //                                     |
        //                                     |
        // (gradient)->reshape---|             |
        //                       V             V
        //(gradient)->reshape-> concat   ->ApplyGradient-> Result
        std::shared_ptr<GNode> allreduce_node;
        auto allreduce_op = std::make_shared<AllReduce>();
        allreduce_node = graph->add_node_and_edge(allreduce_op, {concat_node});
        NNFUSION_LOG(INFO) << "[sc dp pass] allreduce name:" << allreduce_node->get_name();
        // Weight(weight_node) ----------------------------|
        //                                                 |
        //                                                 |
        // (gradient)->reshape---|                         |
        //                       V                         V
        //(gradient)->reshape-> concat --> allreduce   ->ApplyGradient-> Result
        std::vector<std::pair<std::shared_ptr<GNode>, int>> allreduced_gradients_key =
            split_from_one(graph, hash_to_gradient_apply, allreduce_node, subgroup);
        // Weight(weight_node) --------------------------------------------------------------------|
        //                                                                                         |
        //                                                                                         |
        // (gradient)->reshape---|                                                                 V
        //                       | |->reshape->ApplyGradient-> Result
        //                       V                                              |
        //(gradient)->reshape-> concat(concated_gradient) --> allreduce -->
        //slice->reshape->ApplyGradient-> Result

        for (std::pair<std::shared_ptr<GNode>, int> reshape_key : allreduced_gradients_key)
        {
            std::shared_ptr<GNode> apply_node = hash_to_gradient_apply[reshape_key.second].second;
            int gradient_index = get_gradient_from_apply(apply_node);
            graph->remove_edge(apply_node->get_in_edge(gradient_index));
            graph->add_edge(reshape_key.first, 0, apply_node, gradient_index);
        }
    }
    return true;
}

bool SuperScalerDataParallelismPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    sc_allreduce_enable = FLAGS_fadd_sc_allreduce;
    sc_allreduce_fusion_enable = FLAGS_fadd_sc_allreduce_fusion;
    if (!sc_allreduce_enable)
        return true;
    std::map<int, std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>> hash_to_gradient_apply;
    // group gradient and apply* op from n-th layer to 1st layer
    for (int i = graph->get_outputs().size() - 1; i >= 0; i--)
    {
        auto result_node = graph->get_outputs()[i];
        // the apply node followed by result node. so check result_node's input node
        int gradient_index = get_gradient_from_apply((result_node->get_in_edge(0)->get_src()));
        if (gradient_index == -1)
            continue; // skip nodes whose type are not Apply*.
        NNFUSION_CHECK(result_node->get_in_edges().size() == 1)
            << "result node has other input except apply op:";
        // Weight(weight_node) ----|
        //                         |
        //                         V
        // (gradient) --------> Apply-> Result
        auto apply_node = result_node->get_in_edge(0)->get_src();
        std::shared_ptr<GNode> gradient_node = apply_node->get_in_edge(gradient_index)->get_src();
        NNFUSION_LOG(INFO) << "[sc dp pass] find gradient: " << gradient_node->get_name()
                           << "; id: " << hash_to_gradient_apply.size();
        hash_to_gradient_apply[hash_to_gradient_apply.size()] =
            std::pair<std::shared_ptr<GNode>, std::shared_ptr<GNode>>(gradient_node, apply_node);
    }
    if (sc_allreduce_fusion_enable)
    {
        return add_fused_allreduce(graph, hash_to_gradient_apply);
    }
    else
    {
        return add_allreduce(graph, hash_to_gradient_apply);
    }
}
