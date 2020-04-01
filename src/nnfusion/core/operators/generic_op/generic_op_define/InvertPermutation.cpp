// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

REGISTER_OP(InvertPermutation)
    .attr<nnfusion::op::OpConfig::any>("T")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // enforce is like assert, but when thing goes wrong, it will print error message.
        CHECK(gnode->get_input_size() == 1)
            << "Only one input is allowed for the InvertPermutation operator";

        auto& shape_0 = gnode->get_input_shape(0);
        CHECK(shape_0.size() == 1) << "The input only can take a 1-D integer tensor";

        auto ng_op = gnode->get_in_edge(0)->get_src();
        if (ng_op->get_op_type() == "Constant")
        {
            CHECK(gnode->get_input_element_type(0) == nnfusion::element::i32 ||
                  gnode->get_input_element_type(0) == nnfusion::element::i64);
            std::unordered_map<int64_t, int64_t> element_records;
            std::vector<int64_t> input_vector;
            if (gnode->get_input_element_type(0) == nnfusion::element::i32)
            {
                auto temp = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                                ->get_vector<int>();
                input_vector.insert(input_vector.begin(), temp.begin(), temp.end());
            }
            else
            {
                input_vector =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                        ->get_vector<int64_t>();
            }

            for (int i = 0; i < input_vector.size(); i++)
            {
                CHECK(input_vector[i] >= 0 && input_vector[i] < input_vector.size())
                    << "The elements for InvertPermutation's inputs must between 0 to n-1";
                element_records[input_vector[i]]++;
                CHECK(element_records[input_vector[i]] == 1)
                    << "The frequency of a number in InvertPermutation's inputs cannot above 1";
            }
        }

        nnfusion::Shape output_shape_0(shape_0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });