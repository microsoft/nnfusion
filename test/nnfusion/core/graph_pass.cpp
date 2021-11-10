// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"

#include "../test_util/common.hpp"

DECLARE_string(fconst_folding_backend);

const shared_ptr<Graph> generate_test_graph()
{
}

TEST(nnfusion_core, constant_folding_pass)
{
    using namespace nnfusion::pass::graph;
    FLAGS_fconst_folding_backend = "CUDA";

    Shape shape{4};
    std::vector<int> values_a(4, 10);
    std::vector<int> values_b(4, 22);
    std::vector<int> values_c(4, 31);
    auto graph = make_shared<Graph>("constant_folding_pass_without_output");

    Shape shape_a{2, 3};
    auto a = make_shared<op::Constant>(element::i32, shape, values_a);
    auto a_gnode = graph->add_node_and_edge(a, GNodeVector({}));
    auto b = make_shared<op::Constant>(element::i32, shape, values_b);
    auto b_gnode = graph->add_node_and_edge(b, GNodeVector({}));
    auto c = make_shared<op::Constant>(element::i32, shape, values_c);
    auto c_gnode = graph->add_node_and_edge(c, GNodeVector({}));

    auto op1 = std::make_shared<op::Add>();
    auto op2 = std::make_shared<op::Add>();
    auto op_gnode_1 = graph->add_node_and_edge(op1, {a_gnode, b_gnode});
    auto op_gnode_2 = graph->add_node_and_edge(op2, {op_gnode_1, c_gnode});

    auto node_size_before = graph->get_node_size();

    auto const_folding_optimizer = RuntimeConstantFoldingPass();
    const_folding_optimizer.run_on_graph(graph);

    auto result_op =
        std::dynamic_pointer_cast<op::Constant>(graph->get_nodes().at(0)->get_op_ptr());
    auto constant_values = result_op->get_value_strings();

    auto node_size_after = graph->get_node_size();
    EXPECT_EQ(node_size_before, 5);
    EXPECT_EQ(node_size_after, 1);
    EXPECT_TRUE(graph->get_nodes().at(0)->is_constant());
    EXPECT_EQ(constant_values.size(), 4);
    EXPECT_EQ(constant_values.at(0), "63");

    auto graph_wo = make_shared<Graph>("constant_folding_pass_with_output");
    auto a_gnode_wo = graph_wo->add_node_and_edge(a, GNodeVector({}));
    auto b_gnode_wo = graph_wo->add_node_and_edge(b, GNodeVector({}));
    auto c_gnode_wo = graph_wo->add_node_and_edge(c, GNodeVector({}));
    auto op_gnode_wo_1 = graph_wo->add_node_and_edge(op1, {a_gnode_wo, b_gnode_wo});
    auto op_gnode_wo_2 = graph_wo->add_node_and_edge(op2, {op_gnode_wo_1, c_gnode_wo});
    graph_wo->set_default_outputs();
    auto node_size_before_wo = graph_wo->get_node_size();
    const_folding_optimizer.run_on_graph(graph_wo);
    auto node_size_after_wo = graph_wo->get_node_size();
    EXPECT_EQ(node_size_before_wo, 5);
    EXPECT_EQ(node_size_after_wo, 3);
}
