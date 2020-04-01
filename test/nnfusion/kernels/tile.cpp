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

TEST(nnfusion_core_kernels, tile)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_b{1};
    vector<int64_t> multiples{2};
    auto B = make_shared<op::Constant>(element::i64, shape_b, multiples);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B};

    string node_type("Tile");
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);
    // Prepare test data
    auto IN = vector<int>{/*A*/ 1, 2, 3, 4, /*B*/ 2};
    auto OUT = vector<int>{1, 2, 3, 4, 1, 2, 3, 4};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}
