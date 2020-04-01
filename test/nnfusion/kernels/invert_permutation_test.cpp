// Microsoft (c) 2019, MSRA/NNFUSION Team

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

TEST(nnfusion_core_kernels, InvertPermutation)
{
    Shape shape_a{5};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A};

    string node_type("InvertPermutation");

    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);

    auto IN = vector<int>{3, 4, 0, 2, 1};
    auto OUT = vector<int>{2, 4, 3, 0, 1};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}