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

TEST(nnfusion_core_kernels, unsorted_segment_sum)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_b{3};
    vector<int> b{0, 1, 0};
    auto B = make_shared<op::Constant>(element::i32, shape_b, b);
    Shape shape_c{1};
    vector<int> c{2};
    auto C = make_shared<op::Constant>(element::i32, shape_c, c);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B, C};

    string node_type("UnsortedSegmentSum");
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);

    // Prepare test data
    auto IN = vector<int>{/*A*/ 1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, /*B*/ 0, 1, 0, /*C*/ 2};
    auto OUT = vector<int>{5, 5, 5, 5, 5, 6, 7, 8};
    EXPECT_TRUE(nnfusion::test::check_kernel(gnode, CUDA_GPU, IN, OUT));
}
