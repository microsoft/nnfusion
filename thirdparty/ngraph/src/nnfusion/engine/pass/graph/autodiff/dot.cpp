//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"

shared_ptr<op::Reshape>
    make_reshape_axes_to_front(const Shape& front_shape, const Shape& back_shape, bool trans)
{
    AxisVector input_order;
    Shape output_shape;

    for (size_t i = 0; i < back_shape.size(); i++)
    {
        input_order.push_back(front_shape.size() + i);
        output_shape.push_back(back_shape[i]);
    }

    for (size_t i = 0; i < front_shape.size(); i++)
    {
        input_order.push_back(i);
        output_shape.push_back(front_shape[i]);
    }

    return make_shared<op::Reshape>(input_order, output_shape);
}

REGISTER_BACKWARD_TRANSLATOR(Dot).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        //  y = dot(a, b), a_grad = dot(y_grad, b^t), b_grad = dot(a^t, y_grad)
        NNFUSION_CHECK(outputs_grad.size() == 1) << "parameter have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto dot_op = std::dynamic_pointer_cast<op::Dot>(forward_node->get_op_ptr());
        auto reduction_axes_count = dot_op->get_reduction_axes_count();
        auto tran_a = dot_op->get_transpose_A();
        auto tran_b = dot_op->get_transpose_B();
        auto a = get_node_input(forward_node, 0);
        auto b = get_node_input(forward_node, 1);
        if (tran_a)
        {
            auto tran_gnode =
                nnfusion::graph::numpy_transpose(a.gnode, nnfusion::AxisVector(), a.index);
            graph->add_node(tran_gnode);
            graph->add_edge(a.gnode, a.index, tran_gnode, 0);
            a = GNodeIndex{tran_gnode, 0};
        }
        if (tran_b)
        {
            auto tran_gnode =
                nnfusion::graph::numpy_transpose(b.gnode, nnfusion::AxisVector(), b.index);
            graph->add_node(tran_gnode);
            graph->add_edge(b.gnode, b.index, tran_gnode, 0);
            b = GNodeIndex{tran_gnode, 0};
        }

        auto a_shape = a.get_shape();                     // IJ
        auto b_shape = b.get_shape();                     // JK
        auto y_shape = forward_node->get_output_shape(0); // IK
        Shape I_shape, J_shape, K_shape;

        I_shape.insert(I_shape.begin(), a_shape.begin(), a_shape.end() - reduction_axes_count);
        J_shape.insert(J_shape.begin(), b_shape.begin(), b_shape.begin() + reduction_axes_count);
        K_shape.insert(K_shape.begin(), b_shape.begin() + reduction_axes_count, b_shape.end());

        auto bt_op = make_reshape_axes_to_front(J_shape, K_shape, false);
        auto bt_gnode = graph->add_node_and_edge(bt_op, {b});
        auto a_grad = graph->add_node_and_edge(std::make_shared<op::Dot>(K_shape.size()),
                                               {outputs_grad[0], GNodeIndex{bt_gnode, 0}});
        if (tran_a)
        {
            auto tran_gnode = nnfusion::graph::numpy_transpose(a_grad, nnfusion::AxisVector(), 0);
            graph->add_node(tran_gnode);
            graph->add_edge(a_grad, 0, tran_gnode, 0);
            a_grad = tran_gnode;
        }

        auto at_op = make_reshape_axes_to_front(I_shape, J_shape, false);
        auto at_gnode = graph->add_node_and_edge(at_op, {a});
        auto b_grad = graph->add_node_and_edge(std::make_shared<op::Dot>(I_shape.size()),
                                               {GNodeIndex{at_gnode, 0}, outputs_grad[0]});
        if (tran_b)
        {
            auto tran_gnode = nnfusion::graph::numpy_transpose(b_grad, nnfusion::AxisVector(), 0);
            graph->add_node(tran_gnode);
            graph->add_edge(b_grad, 0, tran_gnode, 0);
            b_grad = tran_gnode;
        }

        return GNodeIndexVector{GNodeIndex{a_grad, 0}, GNodeIndex{b_grad, 0}};
    });