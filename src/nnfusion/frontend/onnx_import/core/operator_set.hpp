// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "ngraph/node_vector.hpp"

#include "node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        using Operator = std::function<NodeVector(const Node&)>;
        using OperatorSet = std::unordered_map<std::string, std::reference_wrapper<const Operator>>;

    } // namespace onnx_import

} // namespace ngraph
