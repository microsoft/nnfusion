// Microsoft (c) 2019, MSRA/NNFUSION Team
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
#include "nnfusion/core/operators/pad.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_core_kernels, addn)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_b);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B, C};

    string node_type("AddN");
    // Create node for AddN
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);
    // Prepare test data
    auto IN = vector<float>{/*A*/ 1, 2, 3, 4, 5, 6, /*B*/ 0, 1, 2, 3, 4, 5, /*C*/ 2, 4, 5, 3, 1, 2};
    auto OUT = vector<float>{/*tensor(2, 3)*/ 3, 7, 10, 10, 10, 13};

    EXPECT_TRUE(nnfusion::test::check_kernel<float>(gnode, CUDA_GPU, IN, OUT));
}

TEST(nnfusion_core_kernels, addn_large)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{102424};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{102424};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B};

    string node_type("AddN");
    // Create node for AddN
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);

    // Prepare test data
    auto IN = vector<float>(102424 * 2, 1);
    auto OUT = vector<float>(102424, 2);

    EXPECT_TRUE(nnfusion::test::check_kernel<float>(gnode, CUDA_GPU, IN, OUT));
}
