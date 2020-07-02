// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector softmax(const Node& node);

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
