// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<nnfusion::op::OpConfig::any>("T")
    .attr<nnfusion::op::OpConfig::any>("off_value", 1.0f)
    .attr<nnfusion::op::OpConfig::any>("on_value", 0.0f)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        CHECK(1 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int depth = generic_op->localOpConfig.getRoot()["depth"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        std::string t_str = generic_op->localOpConfig.getRoot()["T"];

        size_t bitwidth = 0;
        bool is_real = false;
        bool is_signed = false;
        bool is_quantized = false;
        string c_type_string = "";
        for (const nnfusion::element::Type* t : nnfusion::element::Type::get_known_types())
        {
            if (t->c_type_string() == t_str)
            {
                bitwidth = t->bitwidth();
                is_real = t->is_real();
                is_signed = t->is_signed();
                is_quantized = t->is_quantized();
                c_type_string = t->c_type_string();
                break;
            }
        }
        nnfusion::element::Type type =
            nnfusion::element::Type(bitwidth, is_real, is_signed, is_quantized, c_type_string);

        if (axis == -1)
            axis = shape_0.size() - 1;
        nnfusion::Shape output_shape_0;
        for (int i = 0; i <= axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        output_shape_0.push_back(depth);
        for (int i = axis + 1; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        gnode->set_output_type_and_shape(0, type, output_shape_0);
    });
