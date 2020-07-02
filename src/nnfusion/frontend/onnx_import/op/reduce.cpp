// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstddef>    // std::size_t
#include <functional> // std::multiplies
#include <iterator>   // std::begin, std::end
#include <numeric>    // std::accumulate

#include "ngraph/shape.hpp"
#include "nnfusion/core/operators/constant.hpp"
#include "nnfusion/core/operators/divide.hpp"

#include "reduce.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector reduce_mean(const Node& node)
                {
                    auto input_shape = node.get_ng_inputs().at(0)->get_shape();
                    auto reduction_axes = reduction::detail::get_reduction_axes(node);
                    std::size_t elem_count_product =
                        std::accumulate(std::begin(reduction_axes),
                                        std::end(reduction_axes),
                                        1UL,
                                        [&input_shape](const std::size_t& a, const std::size_t& b) {
                                            return a * input_shape.at(b);
                                        });
                    auto sum_node = reduction::make_ng_reduction_op<ngraph::op::Sum>(
                        node, node.get_ng_inputs().at(0));
                    auto const_node = std::make_shared<ngraph::op::Constant>(
                        sum_node->get_element_type(),
                        Shape{},
                        std::vector<std::size_t>{elem_count_product});

                    auto broadcasted_const_node =
                        make_broadcast_node(const_node, sum_node->get_shape());
                    return {std::make_shared<ngraph::op::Divide>(sum_node, broadcasted_const_node)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
