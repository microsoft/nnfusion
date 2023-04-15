#include "dot_permutation_pass.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/util/util.hpp"
#include "gflags/gflags.h"

#include <queue>

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(fdot_permutation, false, "Enable Dot Permutation Pass");
DEFINE_string(fpermutate_skiplist, "", "List of op types that skips in permutation");

namespace{

    std::unordered_set<std::string> skip_ops = {};
    void parse_skip_ops()
    {
        stringstream ss(FLAGS_fpermutate_skiplist);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            skip_ops.insert(substr);
        }
    }
}

bool DotPermutationPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool using_pass = FLAGS_fdot_permutation;
    if (!using_pass)
        return true;
    parse_skip_ops();

    NNFUSION_LOG(INFO) << "DotPermutationPass::run_on_graph start";
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    size_t kernel_i = 16;
    size_t kernel_j = 16;
    size_t kernel_k = 16;
    for (auto& it : nodes)
    {
        if (skip_ops.count(it->get_op_type()))
            continue;
        if (it->get_op_type() != "Dot" && it->get_op_type() != "BatchMatMul")
        {
            continue;
        }
        if (it->get_op_type() == "Dot"){
                
            // find a dot node
            NNFUSION_LOG(INFO) << "Find a dot node: " << it->get_id();  
            // if node_shape's == 2, continue
            // if (it->get_shape().size() == 2)
            //     continue;
            auto it_op = static_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
            auto trans_a = it_op->get_transpose_A();
            auto trans_b = it_op->get_transpose_B();
            // get the input nodes
            auto input_node = it->get_in_edge(0)->get_src();
            auto weight_node = it->get_in_edge(1)->get_src();
            NNFUSION_LOG(INFO) << "Input node: " << input_node->get_id();
            NNFUSION_LOG(INFO) << "Input Type: " << input_node->get_unique_name();
            NNFUSION_LOG(INFO) << "Weight node: " << weight_node->get_id();
            // if the input_node or weight_node is dot, continue
            if (input_node->get_op_type() == "Dot" || weight_node->get_op_type() == "Dot")
                NNFUSION_LOG(ERROR) << "Currently do not support input node or weight node is dot";
            
            // create a new Permutate Node;
            nnfusion::op::OpConfig::any permutateConfig;
            permutateConfig["type"] = 0;
            permutateConfig["inner_i"] = kernel_i;
            permutateConfig["inner_j"] = kernel_k;
            auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                "Permutate", "Permutate", permutateConfig);
            // convert op to GNode
            auto permutate_node = graph->add_node_and_edge(generic_op, {input_node});
            auto edge = it->get_in_edge(0);
            graph->remove_edge(edge);
            graph->add_edge(permutate_node, 0, it, 0);
            // replace dot with LayoutDot
            nnfusion::op::OpConfig::any layoutDotConfig;
            layoutDotConfig["output_type"] = 0;
            layoutDotConfig["inner_i"] = kernel_i;
            layoutDotConfig["inner_j"] = kernel_j;
            auto layoutDot_op = std::make_shared<nnfusion::op::GenericOp>(
                "LayoutDot", "LayoutDot", layoutDotConfig);
            NNFUSION_LOG(INFO) << "Create layoutDot node";
            // add layoutDot_node behind layout_Dot_op
            NNFUSION_LOG(INFO) << "permutate_node input shape is " << nnfusion::join(permutate_node->get_input_shape(0));
            NNFUSION_LOG(INFO) << "permutate_node shape is "
                            << nnfusion::join(permutate_node->get_shape());

            auto layoutDot_node = graph->add_node_and_edge(layoutDot_op, {permutate_node, weight_node});
            NNFUSION_LOG(INFO) << "Replace it->output's input edge with layoutDot_node";
            for (auto& edge : it->get_out_edges())
            {
                auto dst_node = edge->get_dst();
                auto dst_input = edge->get_dst_input();
                graph->remove_edge(edge);
                graph->add_edge(layoutDot_node, 0, dst_node, dst_input);
            }
            graph->remove_node(it);
            NNFUSION_LOG(INFO) << "Replace dot with layoutDot done";
            // apply layout transform into weight_node
            NNFUSION_LOG(INFO) << "Apply layout transform into weight_node";
            auto weight_shape = weight_node->get_shape();
            auto weight_op = dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
            // assert weight_op != nullptr
            NNFUSION_CHECK_NOT_NULLPTR(weight_op);
            NNFUSION_LOG(INFO) << "weight shape is " << weight_shape[0] << " " << weight_shape[1];
            // get element_type
            auto element_type = weight_op->get_type();
    #define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
    #define OFFSET4D(x, y, z, w, ld1, ld2, ld3) ((x) * (ld1) + (y) * (ld2) + (z) * (ld3) + (w))
            if (element_type == nnfusion::element::f16)
            {
                NNFUSION_LOG(INFO) << "weight_node's element_type is f16";
                // rewrite data as first transpose
                // get data
                half_float::half* data = (half_float::half *)weight_op->get_data_ptr();
                // create a temp storage
                half_float::half* temp_data = (half_float::half*)(new char[weight_op->get_data_size()]);
                // transpose
                // if weight is transposed, direct assign
                if (it_op->get_transpose_B())
                {
                    NNFUSION_LOG(INFO) << "weight_node is transposed";
                    memcpy(temp_data, data, weight_op->get_data_size());
                }
                else
                {
                    NNFUSION_LOG(INFO) << "weight_node is not transposed";

                    for (int i = 0; i < weight_shape[0]; i++)
                    {
                        for (int j = 0; j < weight_shape[1]; j++)
                        {
                            temp_data[OFFSET2D(j, i, weight_shape[0])] =
                                data[OFFSET2D(i, j, weight_shape[1])];
                        }
                    }
                }
            
                // layout transform data[vi / 16, vj / 16, vi % 16, vj % 16] = temp_data[vi / 8 * 8 + vi % 4 * 2 + vj % 16 / 8, vj / 16 * 16 + vi % 8 / 4 * 8 + vj % 8
                for (int i = 0; i < weight_shape[1]; i++)
                {
                    for (int j = 0; j < weight_shape[0]; j++)
                    {
                        data[OFFSET4D(i / 16,
                                    j / 16,
                                    i % 16,
                                    j % 16,
                                    kernel_j * weight_shape[0],
                                    kernel_j * kernel_k,
                                    kernel_k)] =
                            temp_data[OFFSET2D(i / 8 * 8 + i % 4 * 2 + j % 16 / 8,
                                            j / 16 * 16 + i % 8 / 4 * 8 + j % 8,
                                            weight_shape[0])];
                    }
                }
            }
            else{
                NNFUSION_LOG(ERROR) << "weight_node's element_type is not f16";
            }
        }
        else if (it->get_op_type() == "BatchMatMul"){
            NNFUSION_LOG(INFO) << "Find a BatchMatMul node: " << it->get_id();
            // get the input nodes
            auto input_node = it->get_in_edge(0)->get_src();
            auto weight_node = it->get_in_edge(1)->get_src();
            NNFUSION_LOG(INFO) << "Input node: " << input_node->get_id();
            NNFUSION_LOG(INFO) << "Input Type: " << input_node->get_unique_name();
            NNFUSION_LOG(INFO) << "Weight node: " << weight_node->get_id();
            // get node's attr
            auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(it->get_op_ptr());
            bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
            bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

            // Currently do not support constant weight
            NNFUSION_CHECK(weight_node->get_op_type() != "Constant") << "Constant weight is not supported for now";
            // Insert permutate node before BatchMatMul's input node and weight node
            // Insert permutate node before BatchMatMul's input node
            NNFUSION_LOG(INFO) << "Insert permutate node before BatchMatMul's input node";
            {
                // permutate input
                nnfusion::op::OpConfig::any permutateConfig;
                permutateConfig["type"] = trans_A? 1 : 0;
                permutateConfig["inner_i"] = kernel_i;
                permutateConfig["inner_j"] = kernel_k;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "BatchPermutate", "BatchPermutate", permutateConfig);
                auto permutate_node = graph->add_node_and_edge(generic_op, {input_node});
                auto edge = it->get_in_edge(0);
                graph->remove_edge(edge);
                graph->add_edge(permutate_node, 0, it, 0);
            }
            {
                // permutate weight
                nnfusion::op::OpConfig::any permutateConfig;
                permutateConfig["type"] = trans_B ? 1 : 0;
                permutateConfig["inner_i"] = kernel_i;
                permutateConfig["inner_j"] = kernel_k;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "BatchPermutate", "BatchPermutate", permutateConfig);
                auto permutate_node = graph->add_node_and_edge(generic_op, {weight_node});
                auto edge = it->get_in_edge(1);
                graph->remove_edge(edge);
                graph->add_edge(permutate_node, 0, it, 1);
            }
            // replace BatchMatMul with LayoutBMM
            NNFUSION_LOG(INFO) << "Replace BatchMatMul with LayoutBMM";
            {
                nnfusion::op::OpConfig::any layoutBMMConfig;
                layoutBMMConfig["output_type"] = 0;
                layoutBMMConfig["inner_i"] = kernel_i;
                layoutBMMConfig["inner_j"] = kernel_j;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "LayoutBMM", "LayoutBMM", layoutBMMConfig);
                NNFUSION_LOG(INFO) << "Create LayoutBMM node";
                auto layoutBMM_node = graph->add_node_and_edge(
                    generic_op, {it->get_in_edge(0)->get_src(), it->get_in_edge(1)->get_src()});
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(layoutBMM_node, 0, dst_node, dst_input);
                }
                graph->remove_node(it);
            }
        }
    }
#undef OFFSET2D
#undef OFFSET4D
    return true;
}
