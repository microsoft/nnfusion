// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/node.hpp"
#include "ngraph/node_vector.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief Convert ONNX ArgMax operation to an nGraph node.
                ///
                /// \param node   The ONNX node object representing this operation.
                ///
                /// \return The vector containing an Ngraph node which produces the output
                ///         of an ONNX ArgMax operation.
                NodeVector argmax(const Node& node);

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
