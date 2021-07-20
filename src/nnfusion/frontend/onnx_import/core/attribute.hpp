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

#include "../util/util.hpp"
#include "tensor.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace detail
            {
                template <typename T>
                inline T get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK_FAIL()
                        << "unsupported attribute type : "
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());
                }

                template <>
                inline float get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT: return attribute.i();
                    case onnx::AttributeProto_AttributeType_FLOAT: return attribute.f();
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline std::vector<float> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<float>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    case onnx::AttributeProto_AttributeType_FLOAT: return {attribute.f()};
                    case onnx::AttributeProto_AttributeType_FLOATS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline double get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_FLOAT:
                        return static_cast<double>(attribute.f());
                    case onnx::AttributeProto_AttributeType_INT: return attribute.i();
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline std::vector<double> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<double>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    case onnx::AttributeProto_AttributeType_FLOAT:
                        return {static_cast<double>(attribute.f())};
                    case onnx::AttributeProto_AttributeType_FLOATS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline std::size_t get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK(attribute.type() == onnx::AttributeProto_AttributeType_INT)
                        << "invalid attribute type : "
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());

                    return static_cast<std::size_t>(attribute.i());
                }

                template <>
                inline std::vector<std::size_t> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<std::size_t>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline int64_t get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK(attribute.type() == onnx::AttributeProto_AttributeType_INT)
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());

                    return attribute.i();
                }

                template <>
                inline std::vector<int64_t> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT: return {attribute.i()};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline std::string get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK(attribute.type() == onnx::AttributeProto_AttributeType_STRING)
                        << "invalid attribute type : "
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());

                    return attribute.s();
                }

                template <>
                inline std::vector<std::string> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_STRING: return {attribute.s()};
                    case onnx::AttributeProto_AttributeType_STRINGS:
                        return {std::begin(attribute.strings()), std::end(attribute.strings())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline Tensor get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK(attribute.type() == onnx::AttributeProto_AttributeType_TENSOR)
                        << "invalid attribute type : "
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());

                    return Tensor{attribute.t()};
                }

                template <>
                inline std::vector<Tensor> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_TENSOR: return {Tensor{attribute.t()}};
                    case onnx::AttributeProto_AttributeType_TENSORS:
                        return {std::begin(attribute.tensors()), std::end(attribute.tensors())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

                template <>
                inline onnx::GraphProto get_value(const onnx::AttributeProto& attribute)
                {
                    NNFUSION_CHECK(attribute.type() == onnx::AttributeProto_AttributeType_GRAPH)
                        << "invalid attribute type : "
                        << onnx::AttributeProto_AttributeType_Name(attribute.type());

                    return attribute.g();
                }

                template <>
                inline std::vector<onnx::GraphProto>
                    get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_GRAPH:
                        return {onnx::GraphProto{attribute.g()}};
                    case onnx::AttributeProto_AttributeType_GRAPHS:
                        return {std::begin(attribute.graphs()), std::end(attribute.graphs())};
                    default:
                        NNFUSION_CHECK_FAIL()
                            << "invalid attribute type : "
                            << onnx::AttributeProto_AttributeType_Name(attribute.type());
                    }
                }

            } // namespace detail

            class Attribute
            {
            public:
                Attribute() = delete;
                explicit Attribute(const onnx::AttributeProto& attribute_proto)
                    : m_attribute_proto{&attribute_proto}
                {
                }

                Attribute(Attribute&&) noexcept = default;
                Attribute(const Attribute&) = default;

                Attribute& operator=(Attribute&&) noexcept = delete;
                Attribute& operator=(const Attribute&) = delete;

                const std::string& get_name() const { return m_attribute_proto->name(); }
                bool is_tensor() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_TENSOR;
                }
                bool is_tensor_array() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_TENSORS;
                }
                bool is_float() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_FLOAT;
                }
                bool is_float_array() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_FLOATS;
                }
                bool is_integer() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_INT;
                }
                bool is_integer_array() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_INTS;
                }
                bool is_string() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_STRING;
                }
                bool is_string_array() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_STRINGS;
                }
                bool is_graph() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_GRAPH;
                }
                bool is_graph_array() const
                {
                    return m_attribute_proto->type() == onnx::AttributeProto_AttributeType_GRAPHS;
                }
                Tensor get_tensor() const { return Tensor{m_attribute_proto->t()}; }
                float get_float() const { return m_attribute_proto->f(); }
                int64_t get_integer() const { return m_attribute_proto->i(); }
                const std::string& get_string() const { return m_attribute_proto->s(); }
                onnx::GraphProto get_graphproto() const;

                std::vector<Tensor> get_tensor_array() const
                {
                    return {std::begin(m_attribute_proto->tensors()),
                            std::end(m_attribute_proto->tensors())};
                }

                std::vector<float> get_float_array() const
                {
                    return {std::begin(m_attribute_proto->floats()),
                            std::end(m_attribute_proto->floats())};
                }

                std::vector<int64_t> get_integer_array() const
                {
                    return {std::begin(m_attribute_proto->ints()),
                            std::end(m_attribute_proto->ints())};
                }

                std::vector<std::string> get_string_array() const
                {
                    return {std::begin(m_attribute_proto->strings()),
                            std::end(m_attribute_proto->strings())};
                }

                std::vector<onnx::GraphProto> get_graphproto_array() const;

                /* explicit */ operator onnx::AttributeProto_AttributeType() const
                {
                    return m_attribute_proto->type();
                }

                template <typename T>
                T get_value() const
                {
                    return detail::get_value<T>(*m_attribute_proto);
                }

            private:
                const onnx::AttributeProto* m_attribute_proto;
            };

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
