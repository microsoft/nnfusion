// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "ngraph/node.hpp"

#include "transpose.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector transpose(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data = node.get_ng_inputs().at(0);

                    auto permute_axes =
                        node.get_attribute_value<std::vector<std::size_t>>("perm", {});

                    return {(permute_axes.empty()) ? reshape::transpose(data)
                                                   : reshape::reorder_axes(data, permute_axes)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
