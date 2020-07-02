// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attribute.hpp"
#include "graph.hpp"
#include "model.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::vector<Graph> Attribute::get_graph_array(const Model& model) const
        {
            std::vector<Graph> result;
            for (const auto& graph : m_attribute_proto->graphs())
            {
                result.emplace_back(graph, model);
            }
            return result;
        }

        Graph Attribute::get_graph(const Model& model) const
        {
            return Graph{m_attribute_proto->g(), model};
        }

    } // namespace onnx_import

} // namespace ngraph
