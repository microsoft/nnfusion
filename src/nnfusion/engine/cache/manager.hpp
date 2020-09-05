// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <pwd.h>
#include <sqlite3.h>

#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;

namespace nnfusion
{
    namespace cache
    {
        // presently only kernel cache database supported
        // Todo: integrate the interfaces of profiling cache database
        struct kernel
        {
            std::string function;
            std::set<std::string> tags;
            std::map<std::string, float> profile;
            int resource;
        };

        class KernelCacheManager
        {
        public:
            KernelCacheManager();
            ~KernelCacheManager();

            std::vector<kernel> fetch_all(std::string identifier, std::string platform);
            kernel fetch_with_tags(std::string identifier,
                                   std::string platform,
                                   std::set<std::string> tags,
                                   bool efficient = false);
            bool is_valid() { return kernel_cache != nullptr; }
        private:
            std::string m_path;
            static sqlite3* kernel_cache;
        };
    } //namespace cache
} //namespace nnfusion