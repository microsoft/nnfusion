// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DepthwiseConv2dNative)
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .attr<nnfusion::op::OpConfig::any>("padding_type")
    .attr<nnfusion::op::OpConfig::any>("strides")
    .attr<nnfusion::op::OpConfig::any>("dilations")
    .attr<nnfusion::op::OpConfig::any>("padding_before")
    .attr<nnfusion::op::OpConfig::any>("padding_after")
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        if (config["padding_type"] != "SAME")
        {
            return false;
        }
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(gnode->get_input_size() == 2);
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        // [ batch, in_rows, in_cols, in_depth ]
        const Shape& input_shape = gnode->get_input_shape(0);

        // [ filter_rows, filter_cols, in_depth, depth_multiplier]
        const Shape& filter_shape = gnode->get_input_shape(1);

        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nhwc = (data_format == "NHWC");

        const int64_t in_depth = is_nhwc ? input_shape[3] : input_shape[1];
        CHECK(in_depth == filter_shape[2]);
        const int64_t depth_multiplier = filter_shape[3];
        const int64_t out_depth = in_depth * depth_multiplier;
        const int64_t input_rows = is_nhwc ? input_shape[1] : input_shape[2];
        const int64_t input_cols = is_nhwc ? input_shape[2] : input_shape[3];
        const int64_t filter_rows = filter_shape[0];
        const int64_t filter_cols = filter_shape[1];
        const int64_t batch = input_shape[0];

        std::vector<int64_t> strides = op->localOpConfig.getRoot()["strides"];
        CHECK(strides.size() == 2);
        const int64_t out_rows = (input_rows + strides[0] - 1) / strides[0];
        const int64_t out_cols = (input_cols + strides[1] - 1) / strides[1];

        Shape output_shape({batch, out_rows, out_cols, out_depth});

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    });
