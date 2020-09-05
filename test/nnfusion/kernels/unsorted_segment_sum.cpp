// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

///\brief Basic Test example for AddN operator
///
///\author wenxh

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_core_kernels, unsorted_segment_sum)
{
    // Prepare inputs
    auto graph = std::make_shared<graph::Graph>();
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));

    Shape shape_b{3};
    vector<int> b{0, 1, 0};
    auto B = make_shared<op::Constant>(element::i32, shape_b, b);
    auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));

    Shape shape_c{1};
    vector<int> c{2};
    auto C = make_shared<op::Constant>(element::i32, shape_c, c);
    auto C_gnode = graph->add_node_and_edge(C, GNodeVector({}));

    string node_type("UnsortedSegmentSum");
    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, myConfig);
    auto gnode = graph->add_node_and_edge(op, {A_gnode, B_gnode, C_gnode});

    // Prepare test data
    auto IN = vector<int>{/*A*/ 1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, /*B*/ 0, 1, 0, /*C*/ 2};
    auto OUT = vector<int>{5, 5, 5, 5, 5, 6, 7, 8};
    EXPECT_TRUE(nnfusion::test::check_kernel(gnode, CUDA_GPU, IN, OUT));
}
