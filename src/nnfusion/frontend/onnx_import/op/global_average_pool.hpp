// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ngraph/node_vector.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief Convert ONNX GlobalAveragePool operation to an nGraph node.
                ///
                /// \param node   The ONNX node object representing this operation.
                ///
                /// \return The vector containing Ngraph nodes producing output of ONNX GlobalAveragePool
                ///         operation.
                NodeVector global_average_pool(const Node& node);

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
