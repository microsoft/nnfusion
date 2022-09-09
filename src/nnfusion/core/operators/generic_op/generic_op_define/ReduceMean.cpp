// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ReduceMean)
    .attr<std::vector<int64_t>>("axes")
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2(
        [](std::shared_ptr<graph::GNode> gnode) -> std::string
        {
            auto make_layout = [](const std::vector<int>& axes) -> std::string
            {
                std::string ret = "";
                for (auto ax : axes)
                    if (ax != -1)
                        ret += ", N" + std::to_string(ax);
                    else
                        ret += ", NONE";
                return "[" + (axes.empty() ? "N" : ret.substr(2)) + "]";
            };

            NNFUSION_CHECK(gnode->get_input_size() == 1);
            const nnfusion::Shape& input_shape = gnode->get_input_shape(0);
            const size_t input_dims = input_shape.size();
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
            std::vector<int64_t> axes = generic_op->localOpConfig.getRoot()["axes"];

            nnfusion::AxisSet ng_reduction_axes;
            {
                if (axes.empty())
                {
                    auto axes_uint = get_default_order(input_shape);
                    std::copy(axes_uint.begin(),
                              axes_uint.end(),
                              std::inserter(ng_reduction_axes, ng_reduction_axes.end()));
                }
                else
                {
                    for (auto axis : axes)
                    {
                        ng_reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                    }
                }
            }

            std::vector<int> input_ax, output_ax;
            size_t reduced_size = 1;
            for (int i = 0; i < input_shape.size(); ++i)
            {
                if (!ng_reduction_axes.count(i))
                    output_ax.push_back(i);
                else
                {
                    reduced_size *= input_shape[i];
                    output_ax.push_back(-1);
                }
                input_ax.push_back(i);
            }

            nnfusion::Shape oshape;
            for (auto i : output_ax)
            {
                if (i != -1)
                    oshape.push_back(input_shape[output_ax[i]]);
                else
                    oshape.push_back(1);
            }
            gnode->set_output_type_and_shape(0, gnode->get_element_type(), oshape);

            auto expression = "@output0@" + make_layout(output_ax) + " +=! @input0@" +
                              make_layout(input_ax) + " / const(" + std::to_string(reduced_size) +
                              ").cast(" + "@input0@" + make_layout(input_ax) + ".dtype())";
            if (output_ax.empty())
                expression += " where N in 1";
            else
                expression += " where NONE in 1";

            for (int dim_index = 0; dim_index < input_shape.size(); dim_index++)
            {
                expression = expression + ", " + "N" + std::to_string(dim_index) + " in " +
                             to_string(input_shape[dim_index]);
            }

            // FIXME: Need to include annotation
            if (reduced_size == 1L)
                expression += " ## @: memcpy";

            std::cout << expression << "\n";
            return expression;
        });