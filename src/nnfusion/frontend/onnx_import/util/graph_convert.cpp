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

#include "graph_convert.hpp"
#include <sys/stat.h>
#include <type_traits>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "op/custom_op.hpp"
#include "ops_bridge.hpp"

DECLARE_bool(ftraining_mode);

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace
            {
                std::vector<onnx::NodeProto>
                    tp_sort(const std::vector<onnx::NodeProto>& unsorted_nodes,
                            const std::unordered_set<std::string>& external_values)
                {
                    auto randam_string = [](int length = 8) -> string {
                        static string charset =
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
                        string result;
                        result.resize(length);
                        // srand(time(NULL));
                        for (int i = 0; i < length; i++)
                            result[i] = charset[rand() % charset.length()];
                        return result;
                    };

                    std::unordered_map<std::string, std::unordered_set<std::string>>
                        value2uses;                                // value name: used nodes
                    std::unordered_map<std::string, int> indegree; // node name:dedup indegree
                    std::unordered_map<std::string, onnx::NodeProto> name2node;
                    std::vector<onnx::NodeProto> sorted_nodes;
                    for (auto n : unsorted_nodes)
                    {
                        string node_name =
                            (n.has_name() && n.name() != "") ? n.name() : randam_string();
                        NNFUSION_CHECK(name2node.find(node_name) == name2node.end())
                            << "duplicate node name: " << node_name;
                        name2node[node_name] = n;
                        int in = 0;
                        for (auto v : n.input())
                        {
                            auto p = value2uses[v].insert(node_name);
                            if (p.second)
                            {
                                in++;
                            }
                        }
                        indegree[node_name] = in;
                    }

                    for (auto v : external_values)
                    {
                        auto p = value2uses.find(v);
                        if (p == value2uses.end())
                        {
                            NNFUSION_LOG(INFO) << "Unused external_values: " << v;
                            continue;
                        }
                        for (auto use : p->second)
                        {
                            indegree[use]--;
                        }
                    }

                    std::queue<std::string> q;
                    std::unordered_set<std::string> added_nodes;
                    for (auto n : indegree)
                    {
                        if (n.second == 0)
                        {
                            q.push(n.first);
                            added_nodes.insert(n.first);
                        }
                    }
                    for (auto node : added_nodes)
                    {
                        indegree.erase(node);
                    }
                    while (!q.empty())
                    {
                        auto name = q.front();
                        q.pop();
                        auto node_proto = name2node.at(name);
                        sorted_nodes.push_back(node_proto);
                        for (auto v : node_proto.output())
                        {
                            if (v == "")
                                continue;
                            auto p = value2uses.find(v);
                            if (p == value2uses.end())
                            {
                                NNFUSION_LOG(INFO) << "Unused node output, node: " << name
                                                   << ", output: " << v;
                                continue;
                            }
                            for (auto use : p->second)
                            {
                                indegree[use]--;
                                if (indegree[use] == 0)
                                {
                                    q.push(use);
                                    indegree.erase(use);
                                }
                            }
                        }
                    }

                    NNFUSION_CHECK(sorted_nodes.size() == unsorted_nodes.size())
                        << "Illegal graph found. sorted nodes size: " << sorted_nodes.size()
                        << ", unsorted_nodes size: " << unsorted_nodes.size();
                    return sorted_nodes;
                }

                int print_model_proto(const onnx::ModelProto& model_proto)
                {
                    onnx::ModelProto proto_without_init;
                    proto_without_init.CopyFrom(model_proto);
                    proto_without_init.mutable_graph()->mutable_initializer()->Clear();
                    NNFUSION_LOG(INFO) << proto_without_init.DebugString();
                }

                std::string
                    readfile_with_offset_length(std::string path, size_t offset, size_t length)
                {
                    struct stat info;
                    NNFUSION_CHECK((stat(path.c_str(), &info) == 0)) << "path not exists: " << path;
                    ifstream in(path, ifstream::binary);
                    in.seekg(0, in.end);
                    size_t file_length = in.tellg();
                    if (length == 0)
                    {
                        length = file_length - offset;
                    }
                    NNFUSION_CHECK(offset + length <= file_length)
                        << "read exceed file end, " << path << " of size " << file_length
                        << ", read offset " << offset << ", read length " << length;

                    std::string buffer(length, ' ');
                    in.seekg(offset, in.beg);
                    in.read(&buffer[0], length);
                    return buffer;
                }

                void move_external_to_rawdata(onnx::TensorProto& tensor, std::string model_dir)
                {
                    if (tensor.has_data_location() &&
                        tensor.data_location() ==
                            onnx::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL)
                    {
                        NNFUSION_CHECK(tensor.external_data_size() >= 1)
                            << "initializer locate at external data, but no external proto "
                               "provided";
                        string const_file_path = "";
                        size_t offset = 0;
                        size_t length = 0;
                        for (auto i = 0; i < tensor.external_data_size(); i++)
                        {
                            auto& kv_pair = tensor.external_data(i);
                            if (kv_pair.key() == "location")
                            {
                                const_file_path = model_dir + "/" + kv_pair.value();
                            }
                            else if (kv_pair.key() == "offset")
                            {
                                offset = std::stoul(kv_pair.value());
                            }
                            else if (kv_pair.key() == "length")
                            {
                                length = std::stoul(kv_pair.value());
                            }
                            else
                            {
                                NNFUSION_CHECK_FAIL() << "unknown external proto key: "
                                                      << kv_pair.key();
                            }
                        }
                        NNFUSION_CHECK(const_file_path != "")
                            << "no external data location provided";
                        string raw_data =
                            readfile_with_offset_length(const_file_path, offset, length);
                        tensor.clear_data_location();
                        tensor.clear_external_data();
                        tensor.set_raw_data(raw_data);
                    }
                }
            }

            GraphConvert::GraphConvert(const onnx::ModelProto& model_proto,
                                       const std::unordered_map<std::string, size_t>& dim_params,
                                       const string& model_dir)
                : onnx_model_proto{&model_proto}
                , onnx_graph_proto(&(model_proto.graph()))
                , m_graph(new nnfusion::graph::Graph())
                , m_dim_params(dim_params)
                , model_dir(model_dir)
            {
                print_model_proto(model_proto);

                // Note: onnx connect nodes by tensor's name instead of op name
                /*
                ir_version: 3
                producer_name: "ngraph ONNXImporter"
                graph {
                node {
                    input: "A"
                    input: "B"
                    output: "X"
                    name: "add_node1"
                    op_type: "Add"
                }
                node {
                    input: "X"
                    input: "C"
                    output: "Y"
                    name: "add_node2"
                    op_type: "Add"
                }
                name: "test_graph"
                input {
                    name: "A"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                input {
                    name: "B"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                input {
                    name: "C"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                output {
                    name: "Y"
                    type {
                    tensor_type {
                        elem_type: FLOAT
                        shape {
                        dim {
                            dim_value: 1
                        }
                        }
                    }
                    }
                }
                }
                opset_import {
                version: 4
                }
                */
                NNFUSION_LOG(INFO) << "Converting Onnx Graph";
                // Walk through the elements of opset_import field and register operator sets
                // for each domain. An exception UnknownDomain() will raise if the domain is
                // unknown or invalid.
                for (const auto& id : onnx_model_proto->opset_import())
                {
                    m_domain_convert_func_map.emplace(
                        id.domain(),
                        OperatorsBridge::get_convert_func_map(
                            id.version(), (id.domain() == "ai.onnx" ? "" : id.domain())));
                }
                // onnx.proto(.3): the empty string ("") for domain or absence of opset_import field
                // implies the operator set that is defined as part of the ONNX specification.
                const auto dm = m_domain_convert_func_map.find("");
                if (dm == std::end(m_domain_convert_func_map))
                {
                    m_domain_convert_func_map.emplace(
                        "", OperatorsBridge::get_convert_func_map(ONNX_OPSET_VERSION, ""));
                }

                m_graph = std::make_shared<nnfusion::graph::Graph>();

                NNFUSION_CHECK(onnx_graph_proto->sparse_initializer_size() == 0)
                    << "sparse_initializer not supported";

                for (const auto& output : onnx_graph_proto->output())
                {
                    m_output_names.insert(output.name());
                }

                for (auto tensor : onnx_graph_proto->initializer())
                {
                    if (tensor.has_name())
                    {
                        move_external_to_rawdata(tensor, model_dir);
                        if (FLAGS_ftraining_mode)
                        {
                            element::Type type;
                            ONNXDataTypeToNNFusionElementType(
                                static_cast<onnx::TensorProto_DataType>(tensor.data_type()), &type);
                            std::shared_ptr<graph::GNode> input_gnode;
                            auto tensor_op = std::make_shared<op::Parameter>(
                                type,
                                Shape(std::begin(tensor.dims()), std::end(tensor.dims())),
                                false,
                                true);
                            tensor_op->set_name(tensor.name());
                            input_gnode =
                                m_graph->add_node_and_edge(tensor_op, graph::GNodeVector({}));
                            m_node_map[tensor.name()] = {GNodeIndex{input_gnode}};
                        }
                        else
                        {
                            auto tensor_op = make_constant_op(
                                static_cast<onnx::TensorProto_DataType>(tensor.data_type()),
                                Shape(std::begin(tensor.dims()), std::end(tensor.dims())),
                                Tensor{tensor});
                            tensor_op->set_name(tensor.name());
                            auto tensor_gnode =
                                m_graph->add_node_and_edge(tensor_op, graph::GNodeVector({}));
                            m_node_map[tensor.name()] = {GNodeIndex{tensor_gnode}};
                        }
                    }
                }
                // Process all ONNX graph inputs, convert them to NNFusion nodes
                for (const auto& input_proto : onnx_graph_proto->input())
                {
                    ValueInfo input_value_info(input_proto, m_dim_params);
                    std::shared_ptr<graph::GNode> input_gnode;
                    // TODO: parameter might have default value in initializer
                    auto it = m_node_map.find(input_proto.name());
                    if (it != std::end(m_node_map))
                    {
                        NNFUSION_LOG(NNFUSION_WARNING) << "Ignore input: " << input_proto.name()
                                                       << ", because it has a default initializers";
                        NNFUSION_CHECK(it->second.size() == 1)
                            << "Multi outputs found for initializer " << input_proto.name();
                        if (it->second[0].get_element_type() != input_value_info.get_element_type())
                        {
                            auto cast_op =
                                std::make_shared<op::Convert>(input_value_info.get_element_type());
                            cast_op->set_name(input_proto.name());
                            auto input_gnode = m_graph->add_node_and_edge(cast_op, it->second);
                            m_node_map[input_proto.name()] = {GNodeIndex{input_gnode}};
                            if (m_output_names.find(input_gnode->get_name()) !=
                                m_output_names.end())
                            {
                                // TODO: should specify which output of current gnode
                                m_graph_outputs.emplace_back(input_gnode);
                            }
                        }
                    }
                    else
                    {
                        auto input_op = std::make_shared<op::Parameter>(
                            input_value_info.get_element_type(), input_value_info.get_shape());
                        input_op->set_name(input_proto.name());
                        input_gnode = m_graph->add_node_and_edge(input_op, graph::GNodeVector({}));
                        m_node_map[input_proto.name()] = {GNodeIndex{input_gnode}};
                        if (m_output_names.find(input_gnode->get_name()) != m_output_names.end())
                        {
                            // TODO: should specify which output of current gnode
                            m_graph_outputs.emplace_back(input_gnode);
                        }
                    }
                }

                // Verify that ONNX graph contains only nodes of available operator types
                {
                    std::unordered_map<std::string, int64> domain2version;
                    for (const auto& id : onnx_model_proto->opset_import())
                    {
                        if (id.domain() == "com.microsoft.nnfusion.custom")
                        {
                            continue;
                        }
                        domain2version[id.domain() == "ai.onnx" ? "" : id.domain()] = id.version();
                    }
                    std::unordered_set<std::string> unknown_ops;
                    for (const auto& node_proto : onnx_graph_proto->node())
                    {
                        if (!is_operator_available(node_proto))
                        {
                            std::string op =
                                ((node_proto.domain() == "ai.onnx") ? ""
                                                                    : node_proto.domain() + ".") +
                                node_proto.op_type() + ":" +
                                std::to_string(domain2version.at(node_proto.domain()));
                            unknown_ops.insert(op);
                        }
                    }
                    if (unknown_ops.size() > 0)
                    {
                        for (auto op : unknown_ops)
                        {
                            NNFUSION_LOG(ERROR) << "Unsupported op: " << op;
                        }
                        NNFUSION_CHECK_FAIL() << "Unsupported op count: " << unknown_ops.size();
                    }
                }

                // Process ONNX graph nodes, convert to nGraph nodes
                // sorted to avoid non-stardard model
                std::vector<onnx::NodeProto> unsorted_nodes(std::begin(onnx_graph_proto->node()),
                                                            std::end(onnx_graph_proto->node()));
                std::unordered_set<std::string> external_values{
                    ""}; // values provided by initializers/params, empty string means option input
                std::transform(std::begin(onnx_graph_proto->initializer()),
                               std::end(onnx_graph_proto->initializer()),
                               std::inserter(external_values, external_values.begin()),
                               [](onnx::TensorProto t) -> std::string { return t.name(); });
                std::transform(std::begin(onnx_graph_proto->input()),
                               std::end(onnx_graph_proto->input()),
                               std::inserter(external_values, external_values.begin()),
                               [](onnx::ValueInfoProto v) -> std::string { return v.name(); });
                std::vector<onnx::NodeProto> sorted_nodes =
                    tp_sort(unsorted_nodes, external_values);

                graph::GNodeIndexVector optimizer_outputs;

                for (const auto& node_proto : sorted_nodes)
                {
                    auto results = convert_node(node_proto);
                    for (auto& named_gnode : results)
                    {
                        m_node_map[named_gnode.name] = {named_gnode.gnode_index};

                        if (m_output_names.find(named_gnode.name) != m_output_names.end())
                        {
                            // TODO: should specify which output of current gnode
                            named_gnode.gnode_index.gnode->get_output_tensor_ptr(0)->set_name(
                                named_gnode.name);
                            m_graph_outputs.emplace_back(named_gnode.gnode_index.gnode);
                        }
                    }

                    if (node_proto.op_type() == "AdamOptimizer")
                    {
                        optimizer_outputs.emplace_back(results[0].gnode_index);
                    }
                }
                // XX, we hardcode optimizer in output, because onnx training model only output loss
                if (optimizer_outputs.size() > 1)
                {
                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["index"] = 0;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        "sink_node", "SelectNode", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, optimizer_outputs);
                    m_node_map["sink_node"] = {GNodeIndex{generic_gnode}};
                    m_output_names.insert("sink_node");
                    m_graph_outputs.emplace_back(generic_gnode);
                }

                m_graph->set_default_parameters();
                m_graph->set_outputs(m_graph_outputs);

                NNFUSION_LOG(INFO) << "convert graph done";
            }

            NamedNodeVector GraphConvert::convert_node(const onnx::NodeProto& node_proto)
            {
                NNFUSION_LOG(INFO) << "convert node: " << node_proto.name();
                NamedNodeVector ret = get_convert_func(node_proto.op_type(), node_proto.domain())(
                    node_proto, m_node_map, m_graph);
                for (int i = 0; i < ret.size(); i++)
                {
                    NNFUSION_LOG(INFO) << "node " << node_proto.name() << ", output " << ret[i].name
                                       << ", shape " << ret[i].gnode_index.get_shape();
                }
                return std::move(ret);
            }

            const ConvertFunc& GraphConvert::get_convert_func(const std::string& name,
                                                              const std::string& domain) const
            {
                if (domain == "com.microsoft.nnfusion.custom")
                {
                    return custom_translator;
                }
                const auto dm = m_domain_convert_func_map.find(domain);
                NNFUSION_CHECK(dm != std::end(m_domain_convert_func_map)) << "Unknown Domain: "
                                                                          << domain;

                const auto op = dm->second.find(name);
                NNFUSION_CHECK(op != std::end(dm->second))
                    << "Unknown ConvertFunc: " << (domain.empty() ? "" : domain + ".") << name;

                return op->second;
            }

            bool GraphConvert::is_operator_available(const onnx::NodeProto& node_proto) const
            {
                if (node_proto.domain() == "com.microsoft.nnfusion.custom")
                {
                    return true;
                }
                const auto dm = m_domain_convert_func_map.find(node_proto.domain());

                if (dm == std::end(m_domain_convert_func_map))
                {
                    return false;
                }
                const auto op = dm->second.find(node_proto.op_type());
                return (op != std::end(dm->second));
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
