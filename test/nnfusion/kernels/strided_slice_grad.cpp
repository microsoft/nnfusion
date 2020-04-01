// Microsoft (c) 2019, MSRA/NNFUSION Team
///\brief Basic Test example for AddN operator
///
///\author kctang

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

TEST(nnfusion_core_kernels, strided_slice_grad)
{
    // Prepare inputs
    // you can treate both input and weights as ngraph::op::Paramter
    Shape shape_a{3};
    vector<int> x{1, 3, 3};
    auto A = make_shared<op::Constant>(element::i32, shape_a, x);
    Shape shape_b{3};
    auto B = make_shared<op::Parameter>(element::i32, shape_b);
    Shape shape_c{3};
    auto C = make_shared<op::Parameter>(element::i32, shape_c);
    Shape shape_d{3};
    auto D = make_shared<op::Parameter>(element::i32, shape_d);
    Shape shape_e{1, 1, 3};
    auto E = make_shared<op::Parameter>(element::f32, shape_e);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B, C, D, E};

    string node_type("StridedSliceGrad");
    // Create node for StridedSliceGrad
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);
    auto gnode = make_shared<GNode>(node);
    // Prepare test data
    auto IN =
        vector<int>{/*A*/ 1, 3, 3, /*B*/ 0, 0, 0, /*C*/ 0, 1, 0, /*D*/ 1, 1, 1, /*E*/ 4, 5, 6};
    auto OUT = vector<int>{/*tensor(1, 3, 3)*/ 4, 5, 6, 0, 0, 0, 0, 0, 0};

    EXPECT_TRUE(nnfusion::test::check_kernel<int>(gnode, CUDA_GPU, IN, OUT));
}