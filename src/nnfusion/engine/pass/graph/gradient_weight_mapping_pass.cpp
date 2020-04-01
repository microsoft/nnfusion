// Microsoft (c) 2019, NNFusion Team

#include "gradient_weight_mapping_pass.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/allreduce.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/result.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::op;
using namespace nnfusion::pass::graph;

DEFINE_bool(fadd_allreduce, false, "Add Allreduce operater after ApplyGradient operator.");

bool GradientWeightMappingPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool allreduce_enable = FLAGS_fadd_allreduce;
    std::vector<std::shared_ptr<GNode>> result_nodes;

    auto const_nodes = graph->get_const_nodes();

    for (auto node : graph->get_outputs())
    {
        std::shared_ptr<GNode> update_node = node;
        bool is_apply_gradient_op = false;
        if ((*node)["Alias"].is_valid())
        {
            std::string alias = (*node)["Alias"].as<std::string>();
            std::string expected_const_name = alias.substr(alias.find_first_of('/') + 1);
            auto gradient_shape = node->get_output_shape(0);
            std::shared_ptr<GNode> weight_node = nullptr;
            for (auto const_node : const_nodes)
            {
                std::size_t found_pos = const_node->get_name().find("/read/");
                std::string const_node_name = found_pos == std::string::npos
                                                  ? const_node->get_name()
                                                  : const_node->get_name().substr(0, found_pos);
                if (const_node_name == expected_const_name &&
                    const_node->get_output_shape(0) == gradient_shape)
                {
                    weight_node = const_node;
                    break;
                }
            }
            if (weight_node != nullptr)
            {
                OpConfig::any myConfig;
                myConfig["learning_rate"] = 0.001;

                auto const_op = std::dynamic_pointer_cast<Constant>(weight_node->get_op_ptr());
                const_op->is_weight() = true;

                if (allreduce_enable)
                {
                    // Weight(weight_node) -----------|
                    //                                |
                    //                                V
                    // Result(node) -AllReduce-> ApplyGradient-> Parameter
                    auto allreduce_op = std::make_shared<AllReduce>();
                    auto allreduce_gnode = graph->add_node_and_edge(allreduce_op, {node});
                    auto apply_gradient_op = std::make_shared<GenericOp>(
                        "apply_gradient_" + expected_const_name, "ApplyGradient", myConfig);
                    auto apply_gradient_gnode =
                        graph->add_node_and_edge(apply_gradient_op, {weight_node, allreduce_gnode});

                    // weight -> all reduce
                    update_node = apply_gradient_gnode;
                    is_apply_gradient_op = true;
                }
                else
                {
                    auto apply_gradient_op = std::make_shared<GenericOp>(
                        "apply_gradient_" + expected_const_name, "ApplyGradient", myConfig);

                    auto apply_gradient_gnode =
                        graph->add_node_and_edge(apply_gradient_op, {weight_node, node});
                    update_node = apply_gradient_gnode;
                    is_apply_gradient_op = true;
                }
            }
        }

        auto result_op = std::make_shared<Result>();
        if (is_apply_gradient_op)
        {
            result_op->set_needs_copy_to_host(false);
        }
        auto result_node = graph->add_node_and_edge(result_op, {update_node});
        result_nodes.emplace_back(result_node);
    }
    graph->set_outputs(result_nodes);
    return true;
}
