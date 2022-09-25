// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <nnfusion/core/graph/gnode.hpp>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Einsum)
    .infershape([](std::shared_ptr<graph::GNode> curr) -> void {
        auto op = static_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto& cfg = op->localOpConfig.getRoot();
        auto equation = cfg["equation"];
        assert(!equation.is_null());
        NNFUSION_LOG(INFO) << std::string(equation);
        assert(equation == std::string("b i d, b j d -> b i j") || equation == std::string("b i j, b j d -> b i d"));
        if(equation == std::string("b i d, b j d -> b i j"))
        {
          auto input_shape_0 = curr->get_input_shape(0);
          auto input_shape_1 = curr->get_input_shape(1);
          auto output_shape_0 = input_shape_0;
          output_shape_0[2] = input_shape_1[1];
          curr->set_output_type_and_shape(0, curr->get_input_element_type(0), output_shape_0);
          return;
        }
        if(equation == std::string("b i j, b j d -> b i d"))
        {
          auto input_shape_0 = curr->get_input_shape(0);
          auto input_shape_1 = curr->get_input_shape(1);
          auto output_shape_0 = input_shape_0;
          output_shape_0[2] = input_shape_1[2];
          curr->set_output_type_and_shape(0, curr->get_input_element_type(0), output_shape_0);
          return;
        }
        })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto& cfg = op->localOpConfig.getRoot();
        auto equation = cfg["equation"];
        assert(!equation.is_null());
        assert(equation == std::string("b i d, b j d -> b i j") || equation == std::string("b i j, b j d -> b i d"));
        if(equation == std::string("b i d, b j d -> b i j"))
          return "@output0@[B,I,J]+=!@input0@[B,I,D] * @input1@[B,J,D]";
        if(equation == std::string("b i j, b j d -> b i d"))
          return "@output0@[B,I,D]+=!@input0@[B,I,J] * @input1@[B,J,D]";
        return "";
    });
