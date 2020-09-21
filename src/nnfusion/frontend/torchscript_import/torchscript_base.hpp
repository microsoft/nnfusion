//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.
#pragma once

// we need put torch header first to avoid ambiguous log level,
// so disable clang-format changing header order
// clang-format off
#include "torch/script.h"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/frontend/frontend_base.hpp"
// clang-format on

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            using GNodePtr = std::shared_ptr<nnfusion::graph::GNode>;
            using TNodePtr = torch::jit::Node*;
            using TBlockPtr = torch::jit::Block*;
            using TValuePtr = torch::jit::Value*;
            using NodeMap = std::unordered_map<TNodePtr, graph::GNodeVector>;
            using ConvertFunc =
                std::function<graph::GNodeVector(const TNodePtr n,
                                                 NodeMap& tnode2gnodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)>;

        } // namespace torchscript_import
    }     // namespace frontend
} // namespace nnfusion
