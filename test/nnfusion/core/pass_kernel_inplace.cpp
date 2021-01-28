// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/device/reversed_dfs_visitor.hpp"
#include "nnfusion/engine/engine.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::pass::graph;
using namespace std;
using namespace nnfusion::profiler;

class TestEngine : public Engine
{
public:
    TestEngine()
    {
        g_passes->push_back(make_shared<OpInplacePass>());
        // Kernel selection
        g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
        g_passes->push_back(make_shared<DefaultKernelSelector>());

        g_visitor = make_shared<ReversedDFSVisitor>();
    }
};

bool run(std::shared_ptr<nnfusion::graph::Graph> graph)
{
    graph->set_default_outputs();
    TestEngine test_engine;
    test_engine.run_on_graph(graph);

    return true;
}
// bool check_inplace_oi_pair(shared_ptr<nnfusion::op::Op> node)
// {
//     if (auto op = dynamic_pointer_cast<nnfusion::op::Op>(node))
//     {
//         auto annotation = op->get_op_annotations();
//         if (annotation && annotation->get_in_place_oi_pairs().size() > 0)
//         {
//             return true;
//         }
//     }
//     return false;
// }

bool check_inplace_oi_pair(std::shared_ptr<nnfusion::graph::GNode>& node)
{
    auto emitted_kernel =
        (*node)["Kernel_Selection_Result"].as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
    if (emitted_kernel.second->get_or_emit_source() != nullptr)
    {
        KernelEmitter::Pointer kernel = emitted_kernel.second;
        auto annotations = kernel->m_context->annotations;
        if (annotations && annotations->get_in_place_oi_pairs().size() > 0)
            return true;
    }
    return false;
}

TEST(nnfusion_inplace_kernel, reshape)
{
    // Create graph
    std::string name = "Reshape";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    nnfusion::AxisVector input_order{0, 1};
    Shape output_shape{3, 2};

    // Create node
    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(input_order, output_shape);
    auto reshape_gnode = graph->add_node_and_edge(reshape_op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(reshape_gnode));
}

TEST(nnfusion_inplace_kernel, result)
{
    // Create graph
    std::string name = "Result";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Result>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sum)
{
    // Create graph
    std::string name = "Sum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));

    Shape shape_b{2, 3, 1};
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_b);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    nnfusion::AxisSet reduce_axesA;
    nnfusion::AxisSet reduce_axesB{2};

    // Create node
    auto sum_a_op = std::make_shared<nnfusion::op::Sum>(reduce_axesA);
    auto sum_a_gnode = graph->add_node_and_edge(sum_a_op, {para_a_gnode});
    auto sum_b_op = std::make_shared<nnfusion::op::Sum>(reduce_axesB);
    auto sum_b_gnode = graph->add_node_and_edge(sum_b_op, {para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(sum_a_gnode));
    EXPECT_TRUE(check_inplace_oi_pair(sum_b_gnode));
}

TEST(nnfusion_inplace_kernel, broadcast)
{
    // Create graph
    std::string name = "Broadcast";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    Shape shape_b{2, 3};

    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_b);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    nnfusion::AxisSet broadcast_axesA;
    Shape output_shapeA{2, 3};
    nnfusion::AxisSet broadcast_axesB{0};
    Shape output_shapeB{1, 2, 3};

    // Create node
    auto a_op = std::make_shared<nnfusion::op::Broadcast>(output_shapeA, broadcast_axesA);
    auto a_gnode = graph->add_node_and_edge(a_op, {para_a_gnode});
    auto b_op = std::make_shared<nnfusion::op::Broadcast>(output_shapeB, broadcast_axesB);
    auto b_gnode = graph->add_node_and_edge(b_op, {para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(a_gnode));
    EXPECT_TRUE(check_inplace_oi_pair(b_gnode));
}

// TEST(nnfusion_inplace_kernel, max)
// {
//     // Create graph
//     std::string name = "Max";
//     auto graph = std::make_shared<nnfusion::graph::Graph>(name);

// // Prepare inputs
// Shape shape_a{2, 3};
// auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
// auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector{});
// nnfusion::AxisSet reduction_axes;

//     // Create node
//     auto op = std::make_shared<nnfusion::op::Max>(reduction_axes);
//     auto gnode = graph->add_node_and_edge(op, {para_gnode});

//     run(graph);

//     EXPECT_TRUE(check_inplace_oi_pair(gnode));
// }

// TEST(nnfusion_inplace_kernel, min)
// {
//     // Create graph
//     std::string name = "Min";
//     auto graph = std::make_shared<nnfusion::graph::Graph>(name);

//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector{});
//     nnfusion::AxisSet reduction_axes;

// //     // Create node
//     auto op = std::make_shared<nnfusion::op::Min>(reduction_axes);
//     auto gnode = graph->add_node_and_edge(op, {para_gnode});

//     run(graph);

//     EXPECT_TRUE(check_inplace_oi_pair(gnode));
// }

TEST(nnfusion_inplace_kernel, abs)
{
    // Create graph
    std::string name = "Abs";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Abs>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, acos)
{
    // Create graph
    std::string name = "Acos";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Acos>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, asin)
{
    // Create graph
    std::string name = "Asin";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Asin>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, atan)
{
    // Create graph
    std::string name = "Atan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Atan>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, ceiling)
{
    // Create graph
    std::string name = "Ceiling";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Ceiling>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, cos)
{
    // Create graph
    std::string name = "Cos";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Cos>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, cosh)
{
    // Create graph
    std::string name = "Cosh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Cosh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, exp)
{
    // Create graph
    std::string name = "Exp";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Exp>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, floor)
{
    // Create graph
    std::string name = "Floor";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Floor>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, log)
{
    // Create graph
    std::string name = "Log";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Log>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sin)
{
    // Create graph
    std::string name = "Sin";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Sin>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sinh)
{
    // Create graph
    std::string name = "Sinh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Sinh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sqrt)
{
    // Create graph
    std::string name = "Sqrt";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Sqrt>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, tan)
{
    // Create graph
    std::string name = "Tan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Tan>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, tanh)
{
    // Create graph
    std::string name = "Tanh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Tanh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, power)
{
    // Create graph
    std::string name = "Power";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Power>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, subtract)
{
    // Create graph
    std::string name = "Subtract";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Subtract>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, divide)
{
    // Create graph
    std::string name = "Divide";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Divide>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, divnonan)
{
    // Create graph
    std::string name = "DivNoNan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::DivNoNan>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sign)
{
    // Create graph
    std::string name = "Sign";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Sign>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, relu)
{
    // Create graph
    std::string name = "Relu";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Relu>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, negative)
{
    // Create graph
    std::string name = "Negative";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Negative>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, select)
{
    // Create graph
    std::string name = "Select";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::boolean, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));
    auto para_c_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_c_gnode = graph->add_node_and_edge(para_c_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Select>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode, para_c_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, relubackprop)
{
    // Create graph
    std::string name = "ReluBackprop";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::ReluBackprop>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, add)
{
    // Create graph
    std::string name = "Add";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Add>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, addn)
{
    // Create graph
    std::string name = "AddN";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));
    auto para_c_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_c_gnode = graph->add_node_and_edge(para_c_op, GNodeVector({}));

    // Create node
    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<nnfusion::op::GenericOp>(name, name, myConfig);
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode, para_c_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, multiply)
{
    // Create graph
    std::string name = "Multiply";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Multiply>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, minimum)
{
    // Create graph
    std::string name = "Minimum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Minimum>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, maximum)
{
    // Create graph
    std::string name = "Maximum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Maximum>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sigmoid)
{
    // Create graph
    std::string name = "Sigmoid";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::Sigmoid>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

TEST(nnfusion_inplace_kernel, sigmoidbackprop)
{
    // Create graph
    std::string name = "SigmoidBackprop";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, GNodeVector({}));
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, GNodeVector({}));

    // Create node
    auto op = std::make_shared<nnfusion::op::SigmoidBackprop>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(gnode));
}

// TEST(nnfusion_inplace_op, shared_UnaryElementwiseArithmetic)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args = shared_ptr<nnfusion::Node>(A);

//     // Create node
//     auto nodeA = std::make_shared<nnfusion::op::Tan>(args);
//     auto nodeB = std::make_shared<nnfusion::op::Sqrt>(args);

//     // Create graph
//     nnfusion::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A};
//     std::string name = "UnaryElementwiseArithmetic";
//     auto func = make_shared<nnfusion::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }

// TEST(nnfusion_inplace_op, shared_BinaryElementwiseArithmetic)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args0 = shared_ptr<nnfusion::Node>(A);
//     auto B = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args1 = shared_ptr<nnfusion::Node>(B);

//     // Create node
//     auto nodeA = std::make_shared<nnfusion::op::Power>(args0, args1);
//     auto nodeB = std::make_shared<nnfusion::op::Subtract>(args0, args1);

//     // Create graph
//     nnfusion::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A, B};
//     std::string name = "BinaryElementwiseArithmetic";
//     auto func = make_shared<nnfusion::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }

// TEST(nnfusion_inplace_op, shared_select_andn)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::boolean, shape_a);
//     auto args0 = shared_ptr<nnfusion::Node>(A);
//     auto B = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args1 = shared_ptr<nnfusion::Node>(B);
//     auto C = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args2 = shared_ptr<nnfusion::Node>(C);
//     auto inputs = vector<shared_ptr<nnfusion::Node>>{B, C};

//     // Create node
//     string node_type("AddN");
//     auto nodeA = std::make_shared<nnfusion::op::Select>(args0, args1, args2);
//     nnfusion::op::OpConfig::any myConfig;
//     auto nodeB = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, inputs, myConfig);

//     // Create graph
//     nnfusion::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A, B, C};
//     std::string name = "Select_AddN";
//     auto func = make_shared<nnfusion::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }
