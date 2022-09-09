//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "nnfusion/common/symbolic_shape.hpp"
#include "nnfusion/core/graph/graph.hpp"
namespace nnfusion
{
    namespace frontend
    {
        struct ParamInfo
        {
            nnfusion::Shape shape;
            nnfusion::element::Type type;
            ParamInfo(const nnfusion::Shape&, nnfusion::element::Type);
            ParamInfo(const nnfusion::Shape&, const std::string&);
            ParamInfo(const std::string&);
        };

        std::vector<ParamInfo> build_torchscript_params_from_string(const std::string&);

        std::unordered_map<std::string, SymDim> build_onnx_params_from_string(const std::string&);

        std::vector<std::unordered_map<std::string, SymDim>> build_multi_onnx_params_from_string(const std::string& ss);
    } // namespace frontend
} // namespace nnfusion