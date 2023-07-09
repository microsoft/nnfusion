// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"

REGISTER_BACKWARD_TRANSLATOR(Concat).translator([](std::shared_ptr<GNode> forward_node,
                                                   const GNodeIndexVector& outputs_grad,
                                                   std::shared_ptr<nnfusion::graph::Graph> graph)
                                                    -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "Concat have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    auto concat_result = get_node_output(forward_node, 0);

    Shape lower_bounds(concat_result.get_shape().size(), 0);
    Shape upper_bounds = concat_result.get_shape();

    size_t axis =
        std::dynamic_pointer_cast<op::Concat>(forward_node->get_op_ptr())->get_concatenation_axis();
    size_t pos = 0;
    GNodeIndexVector input_grad;
    for (size_t i = 0; i < forward_node->get_input_size(); i++)
    {
        auto input_i_shape = get_node_input(forward_node, i).get_shape();
        auto slice_width = input_i_shape[axis];
        size_t next_pos = pos + slice_width;
        lower_bounds[axis] = pos;
        upper_bounds[axis] = next_pos;
        auto input_i_grad = graph->add_node_and_edge(
            std::make_shared<op::Slice>(lower_bounds, upper_bounds), outputs_grad);
        input_grad.emplace_back(input_i_grad, 0);
        pos = next_pos;
    }

    return input_grad;
});