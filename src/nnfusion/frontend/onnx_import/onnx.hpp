// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <string>

#include "core/operator_set.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // Registers ONNX custom operator
        void register_operator(const std::string& name,
                               std::int64_t version,
                               const std::string& domain,
                               Operator fn);

        /*
        // Convert on ONNX model to a vector of nGraph Functions (input stream)
        std::vector<std::shared_ptr<Function>> load_onnx_model(std::istream&);

        // Convert an ONNX model to a vector of nGraph Functions
        std::vector<std::shared_ptr<Function>> load_onnx_model(const std::string&);

        // Convert the first output of an ONNX model to an nGraph Function (input stream)
        std::shared_ptr<Function> import_onnx_function(std::istream&);

        // Convert the first output of an ONNX model to an nGraph Function
        std::shared_ptr<Function> import_onnx_function(const std::string&);
        */

    } // namespace onnx_import

} // namespace ngraph
