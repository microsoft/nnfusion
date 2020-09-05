// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        class AntaresKEImp
        {
        public:
            using Pointer = shared_ptr<AntaresKEImp>;
            AntaresKEImp() {}
            std::pair<std::string, bool> autogen(const std::string& expr);
            static std::unordered_map<std::string, std::pair<std::string, bool>> code_cache;
        };
    } // namespace kernels
} // namespace nnfusion
