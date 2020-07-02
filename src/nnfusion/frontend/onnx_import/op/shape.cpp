// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

#include "nnfusion/core/operators/constant.hpp"

#include "shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector shape(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto data_shape = data->get_shape();

                    return {std::make_shared<ngraph::op::Constant>(
                        ngraph::element::i64, Shape{data_shape.size()}, data_shape)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
