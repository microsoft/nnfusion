//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once

#include <iostream>
#include <string>

#include "../util/parameter.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        // Convert a TorchScript model to a nnfusion graph
        std::shared_ptr<nnfusion::graph::Graph> load_torchscript_model(const std::string&);

        std::shared_ptr<nnfusion::graph::Graph>
            load_torchscript_model(const std::string&, const std::vector<ParamInfo>&);
    } // namespace frontend
} // namespace nnfusion
