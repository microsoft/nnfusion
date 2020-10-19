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

#include "onnx/onnx-ml.pb.h"

#include "attribute.hpp"
#include "node.hpp"
#include "tensor.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class Node::Impl
            {
            public:
                Impl() = delete;

                Impl(const onnx::NodeProto& node_proto)
                    : m_node_proto{&node_proto}
                    , m_attributes{std::begin(node_proto.attribute()),
                                   std::end(node_proto.attribute())}
                    , m_output_names{std::begin(node_proto.output()), std::end(node_proto.output())}
                {
                }

                const std::vector<Attribute>& attributes() const;

                const std::vector<std::reference_wrapper<const std::string>>&
                    get_output_names() const;

                bool has_attribute(const std::string& name) const;

                template <typename T>
                T get_attribute_value(const std::string& name, T default_value) const;

                template <typename T>
                T get_attribute_value(const std::string& name) const;

                const onnx::NodeProto& node_proto() const;

            private:
                const onnx::NodeProto* m_node_proto;
                std::vector<Attribute> m_attributes;
                std::vector<std::reference_wrapper<const std::string>> m_output_names;
            };

            const onnx::NodeProto& Node::Impl::node_proto() const { return *m_node_proto; }
            const std::vector<Attribute>& Node::Impl::attributes() const { return m_attributes; }
            const std::vector<std::reference_wrapper<const std::string>>&
                Node::Impl::get_output_names() const
            {
                return m_output_names;
            }

            bool Node::Impl::has_attribute(const std::string& name) const
            {
                auto it = std::find_if(
                    std::begin(m_attributes),
                    std::end(m_attributes),
                    [&](const Attribute& attribute) { return attribute.get_name() == name; });
                return !(it == std::end(m_attributes));
            }

            template <typename T>
            T Node::Impl::get_attribute_value(const std::string& name, T default_value) const
            {
                auto it = std::find_if(
                    std::begin(m_attributes),
                    std::end(m_attributes),
                    [&](const Attribute& attribute) { return attribute.get_name() == name; });
                if (it == std::end(m_attributes))
                {
                    return std::forward<T>(default_value);
                }
                return it->template get_value<T>();
            }

            template <typename T>
            T Node::Impl::get_attribute_value(const std::string& name) const
            {
                auto it = std::find_if(
                    std::begin(m_attributes),
                    std::end(m_attributes),
                    [&](const Attribute& attribute) { return attribute.get_name() == name; });
                NNFUSION_CHECK(it != std::end(m_attributes))
                    << "Node (" + m_node_proto->name() + "): unknown attribute \'" + name + "\'";

                return it->template get_value<T>();
            }

            Node::Node(const onnx::NodeProto& node_proto)
                : m_pimpl{new Impl{node_proto}, [](Impl* impl) { delete impl; }}
            {
            }

            Node::Node(Node&& other) noexcept
                : m_pimpl{std::move(other.m_pimpl)}
            {
            }

            Node::Node(const Node& other)
                : m_pimpl{new Impl{other.m_pimpl->node_proto()}, [](Impl* impl) { delete impl; }}
            {
            }

            const std::vector<std::reference_wrapper<const std::string>>&
                Node::get_output_names() const
            {
                return m_pimpl->get_output_names();
            }

            bool Node::has_attribute(const std::string& name) const
            {
                return m_pimpl->has_attribute(name);
            }

            template <>
            float Node::get_attribute_value(const std::string& name, float default_value) const
            {
                return m_pimpl->template get_attribute_value<float>(name, default_value);
            }

            template <>
            double Node::get_attribute_value(const std::string& name, double default_value) const
            {
                return m_pimpl->template get_attribute_value<double>(name, default_value);
            }

            template <>
            std::int64_t Node::get_attribute_value(const std::string& name,
                                                   std::int64_t default_value) const
            {
                return m_pimpl->template get_attribute_value<std::int64_t>(name, default_value);
            }

            template <>
            std::string Node::get_attribute_value(const std::string& name,
                                                  std::string default_value) const
            {
                return m_pimpl->template get_attribute_value<std::string>(name,
                                                                          std::move(default_value));
            }

            template <>
            Tensor Node::get_attribute_value(const std::string& name, Tensor default_value) const
            {
                return m_pimpl->template get_attribute_value<Tensor>(name,
                                                                     std::move(default_value));
            }

            template <>
            onnx::GraphProto Node::get_attribute_value(const std::string& name,
                                                       onnx::GraphProto default_value) const
            {
                return m_pimpl->template get_attribute_value<onnx::GraphProto>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<float> Node::get_attribute_value(const std::string& name,
                                                         std::vector<float> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<float>>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<double> Node::get_attribute_value(const std::string& name,
                                                          std::vector<double> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<double>>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<std::int64_t>
                Node::get_attribute_value(const std::string& name,
                                          std::vector<std::int64_t> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<std::size_t>
                Node::get_attribute_value(const std::string& name,
                                          std::vector<std::size_t> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<Tensor> Node::get_attribute_value(const std::string& name,
                                                          std::vector<Tensor> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<Tensor>>(
                    name, std::move(default_value));
            }

            template <>
            std::vector<onnx::GraphProto>
                Node::get_attribute_value(const std::string& name,
                                          std::vector<onnx::GraphProto> default_value) const
            {
                return m_pimpl->template get_attribute_value<std::vector<onnx::GraphProto>>(
                    name, std::move(default_value));
            }

            template <>
            float Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<float>(name);
            }

            template <>
            double Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<double>(name);
            }

            template <>
            std::int64_t Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::int64_t>(name);
            }

            template <>
            std::size_t Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::size_t>(name);
            }

            template <>
            std::string Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::string>(name);
            }

            template <>
            Tensor Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<Tensor>(name);
            }

            template <>
            onnx::GraphProto Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<onnx::GraphProto>(name);
            }

            template <>
            std::vector<float> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<float>>(name);
            }

            template <>
            std::vector<double> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<double>>(name);
            }

            template <>
            std::vector<std::int64_t> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name);
            }

            template <>
            std::vector<std::size_t> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name);
            }

            template <>
            std::vector<std::string> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<std::string>>(name);
            }

            template <>
            std::vector<Tensor> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name);
            }

            template <>
            std::vector<onnx::GraphProto> Node::get_attribute_value(const std::string& name) const
            {
                return m_pimpl->template get_attribute_value<std::vector<onnx::GraphProto>>(name);
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
