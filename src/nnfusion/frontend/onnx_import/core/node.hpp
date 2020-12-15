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

#include <string>
#include "../util/util.hpp"

namespace onnx
{
    // forward declaration
    class NodeProto;
}

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class Node
            {
            public:
                Node() = delete;
                Node(const onnx::NodeProto& node_proto);

                Node(Node&&) noexcept;
                Node(const Node&);

                Node& operator=(Node&&) noexcept = delete;
                Node& operator=(const Node&) = delete;

                const std::vector<std::reference_wrapper<const std::string>>&
                    get_output_names() const;

                bool has_attribute(const std::string& name) const;

                template <typename T>
                T get_attribute_value(const std::string& name, T default_value) const;

                template <typename T>
                T get_attribute_value(const std::string& name) const;

            private:
                class Impl;
                // In this case we need custom deleter, because Impl is an incomplete
                // type. Node's are elements of std::vector. Without custom deleter
                // compilation fails; the compiler is unable to parameterize an allocator's
                // default deleter due to incomple type.
                std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
            };
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
