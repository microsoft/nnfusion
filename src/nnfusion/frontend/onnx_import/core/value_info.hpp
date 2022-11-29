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
#include "nnfusion/common/symbolic_shape.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class ValueInfo
            {
            public:
                ValueInfo(ValueInfo&&) = default;
                ValueInfo(const ValueInfo&) = default;

                ValueInfo() = delete;
                explicit ValueInfo(
                    const onnx::ValueInfoProto& value_info_proto,
                    const std::unordered_map<std::string, SymDim>& value_dim_params = {})
                    : m_value_info_proto{&value_info_proto}
                {
                    NNFUSION_CHECK(value_info_proto.type().has_tensor_type())
                        << "value info has no tensor type specified.";
                    auto sym_shape = std::make_shared<SymShape>();
                    for (const auto& dim : value_info_proto.type().tensor_type().shape().dim())
                    {
                        if (dim.value_case() ==
                            onnx::TensorShapeProto_Dimension::ValueCase::kDimValue)
                        {
                            m_shape.emplace_back(static_cast<Shape::value_type>(dim.dim_value()));
                            sym_shape->emplace_back(
                                SymDim(static_cast<Shape::value_type>(dim.dim_value())));
                        }
                        else if (dim.value_case() ==
                                 onnx::TensorShapeProto_Dimension::ValueCase::kDimParam)
                        {
                            std::string value_name = dim.dim_param();
                            NNFUSION_CHECK(value_dim_params.count(value_name))
                                << "unknown input dim_param: " << value_name;
                            m_shape.emplace_back(static_cast<Shape::value_type>(
                                value_dim_params.at(value_name).max()));
                            sym_shape->emplace_back(value_dim_params.at(value_name));
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL() << "input dim unset";
                        }
                    }
                    NNFUSION_CHECK(m_value_info_proto->type().tensor_type().has_elem_type())
                        << "value info has no element type specified.";
                    m_type = ONNXDataTypeToNNFusionElementType(onnx::TensorProto_DataType(
                        m_value_info_proto->type().tensor_type().elem_type()));

                    if (sym_shape->is_dynamic())
                    {
                        m_shape.sym_shape = sym_shape;
                    }

                    this->set_sym_shape(sym_shape);
                }

                ValueInfo& operator=(const ValueInfo&) = delete;
                ValueInfo& operator=(ValueInfo&&) = delete;

                const std::string& get_name() const { return m_value_info_proto->name(); }
                const Shape& get_shape() const { return m_shape; }
                const element::Type& get_element_type() const { return m_type; }
                void set_sym_shape(std::shared_ptr<SymShape> sym_shape)
                {
                    this->sym_shape = sym_shape;
                }
                std::shared_ptr<SymShape> get_sym_shape() { return this->sym_shape; }
            private:
                const onnx::ValueInfoProto* m_value_info_proto;
                Shape m_shape;
                element::Type m_type;
                std::shared_ptr<SymShape> sym_shape;
            };

            inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info)
            {
                return (outs << "<ValueInfo: " << info.get_name() << ">");
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
