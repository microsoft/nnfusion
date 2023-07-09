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

TEST(nnfusion_core_kernels, dynamic_stitch)
{
    // Prepare inputs
    auto graph = std::make_shared<graph::Graph>();

    Shape shape_a{2};
    auto indexA = make_shared<op::Constant>(element::i32, shape_a, std::vector<int32_t>{0, 1});
    auto indexA_gnode = graph->add_node_and_edge(indexA, GNodeVector({}));

    auto dataA = make_shared<op::Parameter>(element::f32, shape_a);
    auto dataA_gnode = graph->add_node_and_edge(dataA, GNodeVector({}));

    Shape shape_b{1};
    auto indexB = make_shared<op::Constant>(element::i32, shape_b, std::vector<int32_t>{1});
    auto indexB_gnode = graph->add_node_and_edge(indexB, GNodeVector({}));

    auto dataB = make_shared<op::Parameter>(element::f32, shape_b);
    auto dataB_gnode = graph->add_node_and_edge(dataB, GNodeVector({}));

    string node_type("DynamicStitch");
    // Create node
    nnfusion::op::OpConfig::any myConfig;
    myConfig["N"] = 2;
    auto op = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, myConfig);
    auto gnode =
        graph->add_node_and_edge(op, {indexA_gnode, indexB_gnode, dataA_gnode, dataB_gnode});
    // Prepare test data
    auto IN = vector<float>{/*A*/ 0, 1, /*B*/ 1, /*C*/ 256, 1024, /*D*/ 1};
    auto OUT = vector<float>{/*tensor(2, 3)*/ 256, 1};

    EXPECT_TRUE(nnfusion::test::check_kernel<float>(gnode, CUDA_GPU, IN, OUT));
}
