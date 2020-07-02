// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <numeric>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/node_vector.hpp"
#include "nnfusion/core/operators/add.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace variadic
        {
            /// \brief Create an nGraph version of an ONNX variadic operation.
            ///        This creates a subgraph with a series of binary operations.
            ///
            /// \tparam T Class of an nGraph binary operation (e.g. Add, Minimum, Maximum)
            /// \param node incoming ONNX opearation
            /// \return nGraph node equivalent of the ONNX operation
            template <class T>
            inline NodeVector make_ng_variadic_op(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};

                // Templated binary operation - Creates Add, Minimum, Maximum, etc.
                auto binary_operation = [](const std::shared_ptr<ngraph::Node>& arg0,
                                           const std::shared_ptr<ngraph::Node>& arg1) {
                    return std::make_shared<T>(arg0, arg1);
                };

                // Create a result node as a series of binary operations
                auto result = std::accumulate(
                    std::next(std::begin(ng_inputs)), // First operand value - the second input
                    std::end(ng_inputs),              // Last value - final input
                    ng_inputs.front(),                // Initial value - first input
                    binary_operation);

                return {result};
            }

        } // namespace variadic

    } // namespace  onnx_import

} // namespace  ngraph
