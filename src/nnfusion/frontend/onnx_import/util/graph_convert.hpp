//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <queue>
#include <string>
#include <vector>

#include "../core/node.hpp"
#include "../core/tensor.hpp"
#include "../core/value_info.hpp"
#include "../onnx_base.hpp"

#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class GraphProtoConvert
            {
            public:
                GraphProtoConvert(
                    const onnx::GraphProto& graph_proto,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, size_t>& dim_params = {},
                    const NodeMap& _node_map = NodeMap(),
                    bool flag_subgraph = false);

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_graph; }
                const onnx::GraphProto& get_onnx_proto_graph() const { return *onnx_graph_proto; }
                NamedNodeVector convert_node(const onnx::NodeProto& node_proto);

                /// \brief Access an operator object by its type name and domain name
                /// The function will return the operator object if it exists, or report an error
                /// in case of domain or operator absence.
                /// \param name       type name of the operator object,
                /// \param domain     domain name of the operator object.
                /// \return Reference to the operator object.
                const ConvertFunc& get_convert_func(const std::string& name,
                                                    const std::string& domain) const;

            private:
                const onnx::GraphProto* onnx_graph_proto;

                std::shared_ptr<nnfusion::graph::Graph> m_graph;

                std::unordered_map<std::string, ConvertFuncMap> m_domain_convert_func_map;

                NodeMap m_node_map;

                // TODO: to be removed
                std::set<std::string> m_output_names;

                graph::GNodeVector m_graph_outputs;

                std::unordered_map<std::string, size_t> m_dim_params;
                std::string m_model_dir;

                bool m_flag_subgraph;
            };
            class GraphConvert
            {
            public:
                GraphConvert(const onnx::ModelProto& model_proto,
                             const std::unordered_map<std::string, size_t>& dim_params = {},
                             const string& model_dir = "");

                std::shared_ptr<nnfusion::graph::Graph> get_graph() { return m_graph; }
                const std::string& get_onnx_proto_producer_name() const
                {
                    return onnx_model_proto->producer_name();
                }

                const onnx::GraphProto& get_onnx_proto_graph() const
                {
                    return onnx_model_proto->graph();
                }

                std::int64_t get_onnx_proto_model_version() const
                {
                    return onnx_model_proto->model_version();
                }

                const std::string& get_onnx_proto_producer_version() const
                {
                    return onnx_model_proto->producer_version();
                }

                /// \brief Convert ONNX::GraphProto to nnfusion graph
                /// \param graph_proto ONNX GraphProto
                /// \param _node_map pre-provided node_map, empty by default
                /// \return std::shared_ptr<nnfusion::graph::Graph>
                std::shared_ptr<nnfusion::graph::Graph>
                    convert_graph(const onnx::GraphProto& graph_proto,
                                  const NodeMap& _node_map = NodeMap());

                // /// \brief Construct complete GraphProtos for sub-graphs in control-flow nodes (e.g., If, Loop) by adding the missing information (i.e., inputs) of the GraphProto, which could be processed by GraphProtoConvert to get nnfusion graph
                // /// \param graph_proto the graph_proto of the ONNX model
                // /// \returns unordered_map<controlflow_node.name, onnx::GraphProto>
                // std::unordered_map<std::string, onnx::GraphProto>
                //     construct_controlflow_graphproto(const onnx::GraphProto& graph_proto);

                /// \brief Check availability of operator base on NodeProto.
                /// \return `true` if the operator is available, otherwise it returns `false`.
                bool is_operator_available(const onnx::NodeProto& node_proto) const;

            private:
                const onnx::ModelProto* onnx_model_proto;
                const onnx::GraphProto* onnx_graph_proto;

                std::shared_ptr<nnfusion::graph::Graph> m_graph;

                std::unordered_map<std::string, ConvertFuncMap> m_domain_convert_func_map;

                // std::unordered_map<std::string, onnx::GraphProto> m_controlflow_graphproto_map;

                // NodeMap m_node_map;

                // TODO: to be removed
                // std::set<std::string> m_output_names;

                // graph::GNodeVector m_graph_outputs;

                std::unordered_map<std::string, size_t> m_dim_params;
                std::string m_model_dir;
            };
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
