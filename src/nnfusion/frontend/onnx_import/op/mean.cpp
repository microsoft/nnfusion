// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/add.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/divide.hpp"

#include "mean.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector mean(const Node& node)
                {
                    auto sum = variadic::make_ng_variadic_op<ngraph::op::Add>(node).front();
                    auto shape = sum->get_shape();

                    // Create a Constant representing the number of inputs with the same shape as sum
                    auto count = ngraph::op::Constant::create(
                        sum->get_element_type(),
                        shape,
                        std::vector<int>(shape_size(shape), node.get_ng_inputs().size()));

                    return {sum / count};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
