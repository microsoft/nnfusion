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

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>

#include "onnx_base.hpp"
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            class OperatorsBridge
            {
            public:
                OperatorsBridge(const OperatorsBridge&) = delete;
                OperatorsBridge& operator=(const OperatorsBridge&) = delete;
                OperatorsBridge(OperatorsBridge&&) = delete;
                OperatorsBridge& operator=(OperatorsBridge&&) = delete;

                static ConvertFuncMap get_convert_func_map(std::int64_t version,
                                                           const std::string& domain)
                {
                    return instance()._get_convert_func_map(version, domain);
                }

                static void register_operator(const std::string& name,
                                              std::int64_t version,
                                              const std::string& domain,
                                              ConvertFunc fn)
                {
                    instance()._register_operator(name, version, domain, std::move(fn));
                }

            private:
                std::unordered_map<
                    std::string,
                    std::unordered_map<std::string, std::map<std::int64_t, ConvertFunc>>>
                    m_map;

                OperatorsBridge();

                static OperatorsBridge& instance()
                {
                    static OperatorsBridge instance;
                    return instance;
                }

                void _register_operator(const std::string& name,
                                        std::int64_t version,
                                        const std::string& domain,
                                        ConvertFunc fn);
                ConvertFuncMap _get_convert_func_map(std::int64_t version,
                                                     const std::string& domain);
            };

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
