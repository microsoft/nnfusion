// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Pad)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto op = static_pointer_cast<nnfusion::op::Pad>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::shared_ptr<nnfusion::graph::GNode> pad_value_node = nullptr;
        for (const auto& in_edge : gnode->get_in_edges())
        {
            if (in_edge->get_dst_input() == 1)
            {
                pad_value_node = in_edge->get_src();
                break;
            }
        }

        std::string expression;
        if (auto constant_op =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(pad_value_node->get_op_ptr()))
        {
            auto constant_values = constant_op->get_value_strings();
            NNFUSION_CHECK(1 == constant_values.size());

            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.nn.pad(args("input0"), pad_before=@pad_below@, pad_after=@pad_above@, pad_value=@pad_value@)); )",
                {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
                 {"output_shape", vector_to_string(gnode->get_output_shape(0))},
                 {"pad_below", vector_to_string(op->get_padding_below())},
                 {"pad_above", vector_to_string(op->get_padding_above())},
                 {"pad_value", constant_values[0]}});
        }

        bool pad_zero = true;
        for (auto i : op->get_padding_below())
        {
            if (i != 0)
                pad_zero = false;
        }

        for (auto i : op->get_padding_above())
        {
            if (i != 0)
                pad_zero = false;
        }

        if (pad_zero)
        {
            expression += " ## @annotation: memcpy";
        }

        return expression;
    });
