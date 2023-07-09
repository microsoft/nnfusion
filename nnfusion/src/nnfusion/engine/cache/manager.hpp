// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <sqlite3.h>
#include "nnfusion/common/common.hpp"

namespace nnfusion
{
    namespace cache
    {
        // presently only kernel cache database supported
        // Todo: integrate the interfaces of profiling cache database
        struct KernelEntry
        {
            std::string key;
            std::string identifier;
            std::string op_type;
            nlohmann::json attributes;
            std::string source;
            std::string device_type;
            nlohmann::json function;
            std::set<std::string> tags;
            nlohmann::json miscs;

            std::map<std::string, float> profile;
            int resource;

            KernelEntry()
            {
                key = "";
                identifier = "";
                op_type = "";
                attributes = nlohmann::json();
                source = "";
                device_type = "";
                function = nlohmann::json();
                tags.clear();
                miscs = nlohmann::json();
            }
        };

        using KernelEntry_p = std::shared_ptr<KernelEntry>;

        class KernelCacheManager
        {
        public:
            KernelCacheManager();
            ~KernelCacheManager();

            std::vector<KernelEntry_p> fetch_all(std::string identifier, std::string device_type);
            KernelEntry_p fetch_with_tags(std::string identifier,
                                          std::string device_type,
                                          std::set<std::string> tags,
                                          bool efficient = false);
            std::vector<KernelEntry_p> fetch_with_source(std::string identifier,
                                                         std::string device_type,
                                                         std::string source);
            bool insert_kernel_entry(const KernelEntry_p kernel_entry, bool overwrite = false);
            bool is_valid() { return kernel_cache != nullptr; }
        public:
            // TODO(lingm): SupportOpList depends on the correctness of the KernelContext identifier
            static std::unordered_set<std::string> SupportOpList;

        private:
            std::string m_path;
            static sqlite3* kernel_cache;
        };
    } //namespace cache
} //namespace nnfusion