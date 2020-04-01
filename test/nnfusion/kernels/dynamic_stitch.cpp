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
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_core_kernels, dynamic_stitch)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{2};
    auto indexA = make_shared<op::Constant>(element::i64, shape_a, std::vector<int64_t>{0, 1});
    auto dataA = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_b{1};
    auto indexB = make_shared<op::Constant>(element::i64, shape_b, std::vector<int64_t>{1});
    auto dataB = make_shared<op::Parameter>(element::f32, shape_b);

    auto inputs = vector<shared_ptr<ngraph::Node>>{indexA, indexB, dataA, dataB};

    string node_type("DynamicStitch");
    // Create node
    ngraph::op::OpConfig::any myConfig;
    myConfig["N"] = 2;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);
    // Prepare test data
    auto IN = vector<float>{/*A*/ 0, 1, /*B*/ 1, /*C*/ 256, 1024, /*D*/ 1};
    auto OUT = vector<float>{/*tensor(2, 3)*/ 256, 1};

    // EXPECT_TRUE(nnfusion::test::check_kernel(gnode, CUDA_GPU, IN, OUT));
}
