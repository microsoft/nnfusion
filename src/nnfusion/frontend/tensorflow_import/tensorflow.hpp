//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>

#include "util/graph_convert.hpp"

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace frontend
    {
        // Convert an TensorFlow model to a nnfusion graph (input stream)
        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(std::istream&);

        // Convert an TensorFlow model to a nnfusion graph
        std::shared_ptr<nnfusion::graph::Graph> load_tensorflow_model(const std::string&);
    } // namespace tensorflow_import

} // namespace nnfusion
