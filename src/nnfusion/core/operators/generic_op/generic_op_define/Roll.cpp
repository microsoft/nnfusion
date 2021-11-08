// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Roll)
    .attr<vector<int>>("shifts")
    .attr<vector<size_t>>("dims")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        vector<int> shifts = generic_op->localOpConfig.getRoot()["shifts"];
        vector<size_t> dims = generic_op->localOpConfig.getRoot()["dims"];
        NNFUSION_CHECK_NOT_NULLPTR(generic_op) << "Node type is not "
                                               << gnode->get_op_ptr()->get_op_type();

        NNFUSION_CHECK(gnode->get_input_size() == 1) << "Inputs of Roll operator should be 1.";
        NNFUSION_CHECK(!dims.empty() && !shifts.empty());
        auto input_shape = gnode->get_input_shape(0);

        NNFUSION_CHECK(dims.size() == shifts.size()) << "Roll dims must be the same size as shifts";
        NNFUSION_CHECK(input_shape.size() >= dims.size())
            << "Roll input dimention size must be greater than dims size";

        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        vector<int> shifts = generic_op->localOpConfig.getRoot()["shifts"];
        vector<size_t> dims = generic_op->localOpConfig.getRoot()["dims"];
        std::unordered_map<size_t, int> mp;
        for (size_t i = 0; i < dims.size(); i++)
        {
            mp[dims[i]] = shifts[i];
        }

        auto ir_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@ where @conditions@; )";

        auto output0_shape = curr->get_output_shape(0);
        auto output0_layout = op::create_layout_from_dims(output0_shape);
        std::string input0_layout = "[";
        for (size_t d = 0; d < output0_shape.size(); d++)
        {
            if (mp.find(d) == mp.end())
            {
                input0_layout += output0_layout[d];
            }
            else
            {
                input0_layout += "(" + output0_layout[d] + " + " + output0_layout[d] +
                                 ".val() - (" + to_string(mp[d]) + ")) % " + output0_layout[d] +
                                 ".val()";
            }

            if (d != output0_shape.size() - 1)
            {
                input0_layout += ", ";
            }
            else
            {
                input0_layout += "]";
            }
        }

        std::string conditions;
        for (size_t d = 0; d < output0_shape.size(); d++)
        {
            conditions += output0_layout[d] + " in " + to_string(output0_shape[d]);
            if (d != output0_shape.size() - 1)
                conditions += ", ";
        }

        op::OpConfig::any op_config;
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_layout);
        op_config["input0_layout"] = input0_layout;
        op_config["conditions"] = conditions;

        return op::create_code_from_template(ir_template, op_config);
    });