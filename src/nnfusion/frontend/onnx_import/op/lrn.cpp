// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>

#include "nnfusion/core/operators/lrn.hpp"

#include "core/node.hpp"
#include "lrn.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector lrn(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 1e-4);
                    double beta = node.get_attribute_value<double>("beta", 0.75);
                    double bias = node.get_attribute_value<double>("bias", 1);
                    size_t size = node.get_attribute_value<size_t>("size");

                    return {std::make_shared<ngraph::op::LRN>(data, alpha, beta, bias, size)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
