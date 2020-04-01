// Microsoft (c) 2019, NNFusion Team

#include <memory>

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
// TODO: add StridedSliceGrad
// currently this is a hack impl for BERT_training
REGISTER_OP(StridedSliceGrad)
    .attr<int>("begin_mask", 0)
    .attr<int>("end_mask", 0)
    .attr<int>("ellipsis_mask", 0)
    .attr<int>("new_axis_mask", 0)
    .attr<int>("shrink_axis_mask", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(gnode->get_input_size() == 5);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int begin_mask = generic_op->localOpConfig.getRoot()["begin_mask"];
        int end_mask = generic_op->localOpConfig.getRoot()["end_mask"];
        int ellipsis_mask = generic_op->localOpConfig.getRoot()["ellipsis_mask"];
        int new_axis_mask = generic_op->localOpConfig.getRoot()["new_axis_mask"];
        int shrink_axis_mask = generic_op->localOpConfig.getRoot()["shrink_axis_mask"];
        // TODO: handle the cases that these attrs are not zeros
        CHECK(begin_mask == 0 && end_mask == 0 && ellipsis_mask == 0 && new_axis_mask == 0 &&
              shrink_axis_mask == 0)
            << "do not support mast attributes yet!";

        // Set output size
        auto x_edge = gnode->get_in_edge(0);
        auto x = gnode->get_in_edge(0)->get_src();
        auto x_value = std::dynamic_pointer_cast<nnfusion::op::Constant>(x->get_op_ptr())
                           ->get_vector<int32_t>();
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        int x_size = input_shape_0[0];
        CHECK(x_size == x_value.size());

        //Bert Defaut: nnfusion::Shape output_shape_0 = {1, 256, 1024};
        nnfusion::Shape output_shape_0;
        for (int i = 0; i < x_size; ++i)
            output_shape_0.push_back(x_value[i]);

        gnode->set_output_type_and_shape(0, nnfusion::element::f32, output_shape_0);
    });
