// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"

#include "../test_util/common.hpp"

DECLARE_string(fconst_folding_backend);

TEST(nnfusion_core, constant_folding_pass)
{
    using namespace nnfusion::pass::graph;
    FLAGS_fconst_folding_backend = "CUDA";

    Shape shape{4};
    std::vector<int> values(4, 10);
    auto graph = make_shared<Graph>("constant_folding_pass");

    Shape shape_a{2, 3};
    auto a = make_shared<op::Constant>(element::i32, shape, values);
    auto a_gnode = graph->add_node_and_edge(a, GNodeVector({}));
    auto b = make_shared<op::Constant>(element::i32, shape, values);
    auto b_gnode = graph->add_node_and_edge(b, GNodeVector({}));
    auto c = make_shared<op::Constant>(element::i32, shape, values);
    auto c_gnode = graph->add_node_and_edge(c, GNodeVector({}));

    auto op1 = std::make_shared<op::Add>();
    auto op2 = std::make_shared<op::Add>();
    auto op_gnode_1 = graph->add_node_and_edge(op1, {a_gnode, b_gnode});
    auto op_gnode_2 = graph->add_node_and_edge(op2, {op_gnode_1, c_gnode});

    auto node_size_before = graph->get_node_size();

    auto const_folding_optimizer = RuntimeConstantFoldingPass();
    const_folding_optimizer.run_on_graph(graph);

    auto node_size_after = graph->get_node_size();
    EXPECT_TRUE(node_size_before == 5);
    EXPECT_TRUE(node_size_after == 1);
    EXPECT_TRUE(graph->get_nodes().at(0)->is_constant());
}
