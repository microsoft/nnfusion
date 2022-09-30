// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ArgMax)
    .infershape([](std::shared_ptr<graph::GNode> curr) -> void {
        auto input_shape_0 = curr->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto& cfg = generic_op->localOpConfig.getRoot();
        int axis = cfg["axis"].is_null() ? 0 : int(cfg["axis"]);
        if (axis < 0)
            axis = axis + input_shape_0.size();
        int keepdims = cfg["keepdims"].is_null() ? 1 : int(cfg["keepdims"]);
        int select_last_index =
            cfg["select_last_index"].is_null() ? 0 : int(cfg["select_last_index"]);
        auto output_shape_0 = input_shape_0;
        if (keepdims)
            output_shape_0[axis] = 1;
        else
            output_shape_0.erase(output_shape_0.begin() + axis);
        curr->set_output_type_and_shape(0, element::i64, output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto input_shape_0 = curr->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto& cfg = generic_op->localOpConfig.getRoot();
        int axis = cfg["axis"].is_null() ? 0 : int(cfg["axis"]);
        if (axis < 0)
            axis = axis + input_shape_0.size();
        int keepdims = cfg["keepdims"].is_null() ? 1 : int(cfg["keepdims"]);
        int select_last_index =
            cfg["select_last_index"].is_null() ? 0 : int(cfg["select_last_index"]);
        auto input_layout_0 = op::create_layout_from_dims(input_shape_0);
        auto argmax_dim = input_layout_0[axis];
        std::vector<std::string> exclude_argmax_dim(input_layout_0.begin(), input_layout_0.end());
        exclude_argmax_dim.erase(exclude_argmax_dim.begin() + axis);
        std::vector<std::string> output_layout_0(input_layout_0.begin(), input_layout_0.end());
        std::string dims = "where";
        for (int i = 0; i < input_shape_0.size(); i++)
        {
            if (i > 0)
                dims += ",";
            dims += " " + input_layout_0[i] + " in " + to_string(input_shape_0[i]);
        }
        if (keepdims)
        {
            output_layout_0[axis] = "Z0";
            dims += ", Z0 in 1";
        }
        else
            output_layout_0.erase(output_layout_0.begin() + axis);

        auto ir = op::create_code_from_template(
            "mediate0[@exclude_argmax_dim@] >=! @input0@[@input_layout_0@] @dims@;"
            "@output0@[@output_layout_0@] >=! @argmax_dim@.when([mediate0[@exclude_argmax_dim@] == "
            "@input0@[@input_layout_0@]], const(-1, @argmax_dim@.dtype())).cast(`int64`) @dims@;",
            {{"exclude_argmax_dim", join(exclude_argmax_dim)},
             {"input_layout_0", join(input_layout_0)},
             {"output_layout_0", join(output_layout_0)},
             {"argmax_dim", argmax_dim},
             {"dims", dims}});
        return ir;
    });
