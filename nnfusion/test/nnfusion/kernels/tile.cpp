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

TEST(nnfusion_core_kernels, tile)
{
    // Prepare inputs
    auto graph = std::make_shared<graph::Graph>();
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));

    Shape shape_b{1};
    vector<int64_t> multiples{2};
    auto B = make_shared<op::Constant>(element::i64, shape_b, multiples);
    auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));

    string node_type("Tile");
    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, myConfig);
    auto gnode = graph->add_node_and_edge(op, {A_gnode, B_gnode});
    // Prepare test data
    auto IN = vector<int>{/*A*/ 1, 2, 3, 4, /*B*/ 2};
    auto OUT = vector<int>{1, 2, 3, 4, 1, 2, 3, 4};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}
