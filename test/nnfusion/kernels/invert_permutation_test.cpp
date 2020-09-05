// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_core_kernels, InvertPermutation)
{
    auto graph = std::make_shared<graph::Graph>();

    Shape shape_a{5};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));

    string node_type("InvertPermutation");

    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<op::GenericOp>(node_type, node_type, myConfig);
    auto gnode = graph->add_node_and_edge(op, {A_gnode});

    auto IN = vector<int>{3, 4, 0, 2, 1};
    auto OUT = vector<int>{2, 4, 3, 0, 1};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}