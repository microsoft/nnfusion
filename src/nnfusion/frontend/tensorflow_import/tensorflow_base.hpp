//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once

#include "graph.pb.h"

#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/frontend/frontend_base.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            using NamedNode = std::pair<std::string, std::shared_ptr<nnfusion::graph::GNode>>;
            using NamedNodeVector = std::vector<NamedNode>;
            using NodeMap =
                std::map<std::string, std::vector<std::shared_ptr<nnfusion::graph::GNode>>>;
            using ConvertFunc =
                std::function<NamedNodeVector(const tensorflow::NodeDef&,
                                              const NodeMap&,
                                              std::shared_ptr<nnfusion::graph::Graph> graph)>;

            inline void CopyToArray(const std::string& src, char* dst)
            {
                memcpy(dst, src.data(), src.size());
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
