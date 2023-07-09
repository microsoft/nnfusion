// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        struct AntaresKernelInfo
        {
            std::string kernel_name;
            std::vector<std::string> input_names;
            std::vector<std::string> output_names;
            std::vector<std::string> input_shapes;
            std::vector<std::string> output_shapes;
            std::vector<std::string> input_dtypes;
            std::vector<std::string> output_dtypes;

            using Pointer = std::shared_ptr<AntaresKernelInfo>;
        };

        class AntaresKEImp
        {
        public:
            using Pointer = shared_ptr<AntaresKEImp>;
            AntaresKEImp() {}
            std::pair<std::string, bool> autogen(const std::string& expr);
            static std::unordered_map<std::string, std::pair<std::string, bool>> code_cache;
            static double get_perf(const std::string& response);
            static std::pair<int, int> get_tuning_step(const std::string& response);
            static std::string get_device_name(const std::string& response);
            static std::vector<nnfusion::Shape> get_output_shapes(const std::string& response);
            static std::vector<AntaresKernelInfo::Pointer>
                get_kernel_info(const std::string& response);
        };
    } // namespace kernels
} // namespace nnfusion
