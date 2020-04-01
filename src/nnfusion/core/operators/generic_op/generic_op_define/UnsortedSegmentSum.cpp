// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

/*
Computes a tensor such that \(output[i] = {j...} data[j...]\) where 
the sum is over tuples j... such that segment_ids[j...] == i. 
Unlike SegmentSum, segment_ids need not be sorted and need not cover 
all values in the full range of valid values.
If the sum is empty for a given segment ID i, output[i] = 0. 
If the given segment ID i is negative, the value is dropped and 
will not be added to the sum of the segment.
num_segments should equal the number of distinct segment IDs.
*/

REGISTER_OP(UnsortedSegmentSum).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    CHECK(gnode->get_input_size() == 3) << "Inputs of UnsortedSegmentSum should be 3.";
    // Outshape is as same as input data, (except the first one);
    auto ng_group = gnode->get_in_edge(1)->get_src();
    auto ng_seg = gnode->get_in_edge(2)->get_src();
    CHECK(ng_seg->get_op_type() == "Constant") << "We only accept the sgements number as Constant.";
    auto& shape_0 = gnode->get_input_shape(0);
    auto& shape_1 = gnode->get_input_shape(1);
    auto& shape_2 = gnode->get_input_shape(2);
    auto constop = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_seg->get_op_ptr());
    auto seg_num = constop->get_vector<int>();
    CHECK(shape_0.size() > 0 && shape_1.size() == 1 && seg_num.size() == 1)
        << "Only support 1-D sgments." << shape_0 << shape_1 << shape_2;
    nnfusion::Shape output_shape(shape_0);
    // Output: Has same shape as data,
    // except for the first segment_ids.rank dimensions,
    // which are replaced with a single dimension which has size num_segments.
    output_shape[0] = seg_num[0];
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
});