// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

///\brief Basic Test example for AddN operator
///
///\author kctang

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_core_kernels, strided_slice_grad)
{
    // Prepare inputs
    auto graph = std::make_shared<graph::Graph>();
    Shape shape_a{3};
    vector<int> x{1, 3, 3};
    auto A = make_shared<op::Constant>(element::i32, shape_a, x);
    auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));

    Shape shape_b{3};
    auto B = make_shared<op::Parameter>(element::i32, shape_b);
    auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));

    Shape shape_c{3};
    auto C = make_shared<op::Parameter>(element::i32, shape_c);
    auto C_gnode = graph->add_node_and_edge(C, GNodeVector({}));

    Shape shape_d{3};
    auto D = make_shared<op::Parameter>(element::i32, shape_d);
    auto D_gnode = graph->add_node_and_edge(D, GNodeVector({}));

    Shape shape_e{1, 1, 3};
    auto E = make_shared<op::Parameter>(element::f32, shape_e);
    auto E_gnode = graph->add_node_and_edge(E, GNodeVector({}));

    string node_type("StridedSliceGrad");
    // Create node for StridedSliceGrad
    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<op::GenericOp>(node_type, node_type, myConfig);
    auto gnode = graph->add_node_and_edge(op, {A_gnode, B_gnode, C_gnode, D_gnode, E_gnode});
    // Prepare test data
    auto IN =
        vector<int>{/*A*/ 1, 3, 3, /*B*/ 0, 0, 0, /*C*/ 0, 1, 0, /*D*/ 1, 1, 1, /*E*/ 4, 5, 6};
    auto OUT = vector<int>{/*tensor(1, 3, 3)*/ 4, 5, 6, 0, 0, 0, 0, 0, 0};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}