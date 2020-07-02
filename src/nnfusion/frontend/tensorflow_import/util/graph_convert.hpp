//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "../tensorflow_base.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            class GraphConvert
            {
            public:
                GraphConvert(const tensorflow::GraphDef& proto);

                NamedNodeVector convert_node(const tensorflow::NodeDef& node);

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_graph; }
            private:
                void generate_topology();

                const tensorflow::GraphDef* tf_graph_proto;

                GNodeVector m_graph_outputs;
                GNodeVector m_graph_parameters;

                NodeMap m_node_map;

                std::shared_ptr<nnfusion::graph::Graph> m_graph;

                // node process topology
                std::queue<uint32_t> tf_topology_;
                // pending input count of each node
                std::vector<uint32_t> tf_pending_counts_;
                // the output nodes of each node
                std::vector<std::vector<uint32_t>> tf_node_outputs_;
                // the output node name set
                std::set<std::string> tf_output_name_;
            };
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
