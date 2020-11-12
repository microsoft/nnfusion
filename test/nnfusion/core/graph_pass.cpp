// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"

#include "../test_util/common.hpp"

TEST(nnfusion_core, constant_foldeing_pass)
{
    // using namespace nnfusion::pass::graph;
    //
    // Shape shape{4};
    // std::vector<int> values(4, 10);
    // auto A = make_shared<op::Constant>(element::i32, shape, values);
    // auto B = make_shared<op::Constant>(element::i32, shape, values);
    // auto add = make_shared<op::Add>();

    // auto A_GNode = make_shared<GNode>(A,GNodeVector());
    // auto B_GNode = make_shared<GNode>(B,GNodeVector());
    // auto add_GNode = make_shared<GNode>(add,GNodeVector());

    // std::shared_ptr<Graph> graph;
    // graph->add_node(A_GNode);
    // graph->add_node(B_GNode);
    // graph->add_node(add_GNode);
    // graph->add_edge(A_GNode, 0, add_GNode, 0);
    // graph->add_edge(B_GNode, 0, add_GNode, 1);

    // auto pass_manager = make_shared<nnfusion::GraphPassManager>();
    // pass_manager->push_back(make_shared<RuntimeConstantFoldingPass>());
    // pass_manager->run_on_graph(graph);

    /*
        shared_ptr<runtime::Backend> backend = runtime::Backend::create("INTERPRETER");

    shared_ptr<runtime::interpreter::INTBackend> ibackend =
        static_pointer_cast<runtime::interpreter::INTBackend>(backend);

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, NAN, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 1, 8});
    auto result = backend->create_tensor(element::f32, shape);

    ibackend->set_nan_check(f, true);
    EXPECT_ANY_THROW(ibackend->call_with_validate(f, {result}, {a, b}));
     */
}
