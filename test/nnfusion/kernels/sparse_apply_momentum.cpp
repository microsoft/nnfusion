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

TEST(nnfusion_core_kernels, sparse_apply_momentum)
{
    // Prepare inputs
    auto graph = std::make_shared<graph::Graph>();

    Shape shape_a{3, 2};
    // todo: change op type to variable
    auto var = make_shared<op::Parameter>(element::f32, shape_a);
    auto var_gnode = graph->add_node_and_edge(var, GNodeVector({}));

    auto accum = make_shared<op::Parameter>(element::f32, shape_a);
    auto accum_gnode = graph->add_node_and_edge(accum, GNodeVector({}));

    auto grad = make_shared<op::Parameter>(element::f32, shape_a);
    auto grad_gnode = graph->add_node_and_edge(grad, GNodeVector({}));

    std::vector<int32_t> indices{0, 1, 0};
    string node_type("SparseApplyMomentum");
    // Create node
    nnfusion::op::OpConfig::any myConfig;
    myConfig["use_nesterov"] = true;
    myConfig["lr"] = 0.01;
    myConfig["momentum"] = 0.01;
    myConfig["indices"] = indices;

    auto op = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, myConfig);
    auto gnode = graph->add_node_and_edge(op, {var_gnode, accum_gnode, grad_gnode});
    // Prepare test data
    auto IN = vector<float>{/*var_gnode*/ 1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            /*accum_gnode*/ 0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            /*grad_gnode*/ 0.2,
                            0.1,
                            0.1,
                            0.5,
                            0.3,
                            0.2};
    auto OUT = vector<float>{/*var_gnode*/ 0.99495, 0.99697, 0.99899, 0.99495, 1, 1};

    EXPECT_TRUE(nnfusion::test::check_kernel<float>(gnode, CUDA_GPU, IN, OUT));
}
