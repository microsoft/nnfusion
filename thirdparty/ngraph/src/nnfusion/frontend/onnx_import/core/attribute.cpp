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

#include "attribute.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            std::vector<onnx::GraphProto> Attribute::get_graphproto_array() const
            {
                std::vector<onnx::GraphProto> result;
                for (const auto& graphproto : m_attribute_proto->graphs())
                {
                    result.emplace_back(graphproto);
                }
                return result;
            }

            onnx::GraphProto Attribute::get_graphproto() const { return m_attribute_proto->g(); }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
