// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Resize)
    .attr<std::string>("method")
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        if (config["method"] != "NEAREST")
        {
            return false;
        }
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto input_shape = gnode->get_input_shape(0);

        auto ng_op = gnode->get_in_edge(1)->get_src();
        NNFUSION_CHECK(ng_op->is_constant())
            << "We only accept the Resze input \"scales\" as Constant";
        auto scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                          ->get_vector<float>();
        NNFUSION_CHECK(input_shape.size() == scales.size());

        nnfusion::Shape output_shape(scales.size());
        for (int i = 0; i < scales.size(); i++)
            output_shape[i] = input_shape[i] * scales[i];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto expression_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@ where @cond@; )";

        nnfusion::Shape output_shape = gnode->get_output_shape(0);

        auto ng_op = gnode->get_in_edge(1)->get_src();
        NNFUSION_CHECK(ng_op->is_constant())
            << "We only accept the Tile input \"scales\" as Constant.";
        auto scales = std::dynamic_pointer_cast<nnfusion::op::Constant>(ng_op->get_op_ptr())
                          ->get_vector<float>();

        auto output_layout = op::create_layout_from_dims(output_shape);
        auto input_layout = op::create_layout_from_dims(output_shape);
        std::string cond;
        for (int d = 0; d < input_layout.size(); ++d)
        {
            if (scales[d] > 1)
                input_layout[d] = input_layout[d] + " // " + to_string(int(scales[d]));
            else
                input_layout[d] = input_layout[d] + " * " + to_string(int(1 / scales[d]));
            cond +=
                (cond.empty() ? "" : ", ") + output_layout[d] + " in " + to_string(output_shape[d]);
        }

        auto expr =
            op::create_code_from_template(expression_template,
                                          {{"output0_layout", vector_to_string(output_layout)},
                                           {"input0_layout", vector_to_string(input_layout)},
                                           {"cond", cond}});
        return expr;
    });