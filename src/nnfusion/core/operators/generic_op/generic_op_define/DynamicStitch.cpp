// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

REGISTER_OP(DynamicStitch)
    .attr<int>("N")
    .attr<nnfusion::op::OpConfig::any>("T")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        size_t input_size = gnode->get_input_size();
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int num_partitions = generic_op->localOpConfig.getRoot()["N"];
        CHECK(num_partitions * 2 == input_size);

        bool all_indices_constant = true;
        int64_t max_index = 0;
        nnfusion::Shape output_shape;
        nnfusion::element::Type type;
        std::vector<std::vector<int64_t>> indices_inputs;
        for (int i = 0; i < num_partitions; ++i)
        {
            auto indices_node = gnode->get_in_edge(i)->get_src();
            if (indices_node->get_op_type() == "Constant")
            {
                auto ng_constant_op =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(indices_node->get_op_ptr());
                auto ng_element_type = indices_node->get_element_type();
                CHECK(ng_element_type == nnfusion::element::i64);
                std::vector<int64_t> values;
                values = ng_constant_op->get_vector<int64_t>();

                indices_inputs.push_back(values);
                for (size_t i = 0; i < values.size(); i++)
                {
                    if (values[i] > max_index)
                        max_index = values[i];
                }
            }
            else
            {
                all_indices_constant = false;
                CHECK_FAIL() << "currently we do not support dynamic tensor shape, input_node="
                             << indices_node->get_op_type();
            }
            auto& indices_shape = gnode->get_input_shape(i);
            auto& data_shape = gnode->get_input_shape(i + num_partitions);
            type = gnode->get_input_element_type(i + num_partitions);

            // calculate the sub-shape
            int64_t start = indices_shape.size();
            const int64_t rank = data_shape.size();
            int64_t end = rank;
            if (start > rank)
                start = rank;
            nnfusion::Shape dims;
            dims.push_back(max_index + 1);
            for (int i = start; i < end; i++)
            {
                dims.push_back(data_shape[i]);
            }
            output_shape = dims;
        }
        generic_op->localOpConfig.attr("indices_inputs", indices_inputs);
        gnode->set_output_type_and_shape(0, type, output_shape);
    });
