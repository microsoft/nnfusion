#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/engine/pass/graph/manager.hpp"

#include "../test_util/common.hpp"

TEST(nnfusion_core, constant_foldeing_pass)
{
    /*
        Shape shape{4};
    std::vector<int> values(4, 10); 
    std::shared_ptr<nnfusion::graph::Node> A = make_shared<op::Constant>(element::i32, shape, values);
    std::shared_ptr<nnfusion::graph::Node> B = make_shared<op::Constant>(element::i32, shape, values);
    std::shared_ptr<nnfusion::graph::Node> add = make_shared<op::Add>(A, B);
    std::shared_ptr<nnfusion::graph::Graph> graph;
    graph->add_node(A);
    graph->add_node(B);
    graph->add_node(add);
    graph->add_edge(A, add, 0, 0);
    graph->add_edge(B, add, 0, 1);

    GraphPassManager pass_manager;
    pass_manager.register_pass<ConstantFoldingPass>();
    pass_manager.run_passes(graph);
    
    
     */

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
