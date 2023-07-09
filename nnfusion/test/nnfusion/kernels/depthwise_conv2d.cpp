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

TEST(nnfusion_core_kernels, depthwise_conv2d)
{
    const int depth = 2;
    const int image_width = 2;
    const int image_height = 3;
    const int batch_count = 1;

    auto graph = std::make_shared<graph::Graph>();

    // Prepare inputs
    // The image matrix is ('first/second' channel):
    // | 1/2  |  3/4  |
    // | 5/6  |  7/8  |
    // | 9/10 | 11/12 |
    Shape shape_image{batch_count, image_height, image_width, depth};
    auto image = make_shared<op::Parameter>(element::f32, shape_image);
    auto image_gnode = graph->add_node_and_edge(image, GNodeVector({}));

    // The filter matrix is:
    // | 1/2 |  7/8  | 13/14 |
    // | 3/4 |  9/10 | 15/16 |
    // | 5/6 | 11/12 | 17/18 |
    const int filter_size = 3;
    const int filter_count = 1;
    Shape shape_filter{filter_size, filter_size, depth, filter_count};
    auto filter = make_shared<op::Parameter>(element::f32, shape_filter);
    auto filter_gnode = graph->add_node_and_edge(filter, GNodeVector({}));

    string node_type("DepthwiseConv2dNative");
    // Create node
    std::string tf_padding_type("SAME");
    std::string tf_data_format("NHWC");
    Strides ng_strides({1, 1});
    Strides ng_dilations({1, 1});
    CoordinateDiff ng_padding_below{1, 1};
    CoordinateDiff ng_padding_above{1, 1};

    nnfusion::op::OpConfig::any op_config;
    op_config["data_format"] = tf_data_format;
    op_config["padding_type"] = tf_padding_type;
    op_config["strides"] = ng_strides;
    op_config["dilations"] = ng_dilations;
    op_config["padding_before"] = ng_padding_above;
    op_config["padding_after"] = ng_padding_below;

    auto op = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, op_config);
    auto gnode = graph->add_node_and_edge(op, {image_gnode, filter_gnode});

    // Prepare test data
    auto IN = vector<float>{/*A*/ 1, 2,  3, 4, 5, 6,  7,  8,  9, 10, 11, 12, /*B*/ 1, 2, 7, 8,
                            13,      14, 3, 4, 9, 10, 15, 16, 5, 6,  11, 12, 17,      18};
    auto OUT = vector<float>{/*C*/ 228, 300, 132, 180, 482, 596, 266, 344, 372, 452, 180, 236};

    EXPECT_TRUE(nnfusion::test::check_kernel<float>(gnode, CUDA_GPU, IN, OUT));
}
