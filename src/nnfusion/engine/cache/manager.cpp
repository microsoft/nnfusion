// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "manager.hpp"
#include <limits>
#include <pwd.h>
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_common_ops.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;

DEFINE_string(fkernel_cache_path, "", "Kernel cache DB path");
DEFINE_string(fproduct_name,
              "default",
              "Device product name, like 'GeForce GTX 1080 Ti', 'Tesla V100-PCIE-16GB'");

using namespace nnfusion::cache;

std::unordered_set<std::string> KernelCacheManager::SupportOpList;

sqlite3* KernelCacheManager::kernel_cache = nullptr;
KernelCacheManager::KernelCacheManager()
{
    m_path = (getpwuid(getuid())->pw_dir + std::string("/.cache/nnfusion/kernel_cache.db"));
    if (FLAGS_fkernel_cache_path != "")
    {
        m_path = FLAGS_fkernel_cache_path;
    }
    {
        size_t pos = m_path.find_last_of("/");
        if (pos != std::string::npos && pos != (m_path.size() - 1))
        {
            std::string cache_folder = m_path.substr(0, pos);

            struct stat s;
            if (stat(cache_folder.c_str(), &s) != 0)
            {
                std::string cmd_create_folder = "mkdir -p " + cache_folder;
                system(cmd_create_folder.c_str());
            }
        }
    }

    if (!kernel_cache)
    {
        if (SQLITE_OK == sqlite3_open(m_path.c_str(), &kernel_cache))
        {
            NNFUSION_LOG(INFO) << "Open kernel cache from: " << m_path;
            const char* table_create = R"(
CREATE TABLE IF NOT EXISTS KernelCache(
   Key        TEXT NOT NULL,
   Identifier TEXT NOT NULL,
   OpType     TEXT NOT NULL,
   Attributes TEXT DEFAULT "",
   Source     TEXT DEFAULT "External",
   DeviceType TEXT NOT NULL,
   Function   TEXT NOT NULL,
   Tags       TEXT DEFAULT "",
   Miscs      TEXT DEFAULT "",
   PRIMARY KEY(Key)
   );
)";
            NNFUSION_CHECK(SQLITE_OK == sqlite3_exec(kernel_cache, table_create, NULL, 0, NULL));
        }
        else
        {
            NNFUSION_LOG(ERROR) << "Invalid path to kernel cache: " << m_path << ", "
                                << sqlite3_errmsg(kernel_cache)
                                << ", kernel cache will be disabled";
            kernel_cache = nullptr;
        }
    }

    if (SupportOpList.size() == 0)
    {
        // kernels::cuda::CudaElementOpMap + {"Dot", "Convolution", "AvgPool", "MaxPool", "Fused_Convolution_Relu", "Fused_Convolution_Add_Relu", "Matched_Pattern"}
        for (auto it : kernels::cuda::CudaElementOpMap)
        {
            SupportOpList.insert(it.first);
        }
        SupportOpList.insert({"Dot",
                              "Convolution",
                              "AvgPool",
                              "MaxPool",
                              "Fused_Convolution_Relu",
                              "Fused_Convolution_Add_Relu",
                              "Matched_Pattern"});
    }
}

KernelCacheManager::~KernelCacheManager()
{
    NNFUSION_CHECK(SQLITE_OK == sqlite3_close(kernel_cache));
    kernel_cache = NULL;
}

std::vector<KernelEntry_p> KernelCacheManager::fetch_all(std::string identifier,
                                                         std::string device_type)
{
    NNFUSION_LOG(DEBUG) << "Trying to fetch kernel " << identifier
                        << " on DeviceType: " << device_type;
    sqlite3_stmt* pStmt;
    const char* fetch = R"(
SELECT Key, Identifier, OpType, Attributes, Source, DeviceType, Function, Tags, Miscs FROM KernelCache WHERE (Identifier = ?) AND (DeviceType = ?);
    )";
    NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(kernel_cache, fetch, -1, &pStmt, 0));
    sqlite3_bind_text(pStmt, 1, identifier.data(), identifier.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 2, device_type.data(), device_type.size(), SQLITE_STATIC);

    std::vector<KernelEntry_p> fetched;
    while (SQLITE_ROW == sqlite3_step(pStmt))
    {
        KernelEntry_p fetched_kernel = std::make_shared<KernelEntry>();

        fetched_kernel->key = std::string((char*)sqlite3_column_text(pStmt, 0));
        fetched_kernel->identifier = std::string((char*)sqlite3_column_text(pStmt, 1));
        fetched_kernel->op_type = std::string((char*)sqlite3_column_text(pStmt, 2));
        fetched_kernel->attributes =
            nlohmann::json::parse(std::string((char*)sqlite3_column_text(pStmt, 3)));
        fetched_kernel->source = std::string((char*)sqlite3_column_text(pStmt, 4));
        fetched_kernel->device_type = std::string((char*)sqlite3_column_text(pStmt, 5));
        fetched_kernel->function =
            nlohmann::json::parse(std::string((char*)sqlite3_column_text(pStmt, 6)));
        fetched_kernel->miscs =
            nlohmann::json::parse(std::string((char*)sqlite3_column_text(pStmt, 8)));

        if (SupportOpList.find(fetched_kernel->op_type) == SupportOpList.end())
        {
            NNFUSION_LOG(DEBUG) << "Unsupported op_type: " << fetched_kernel->op_type
                                << ", ingore this fetch";
            fetched.clear();
            break;
        }

        // parse input tags
        size_t pos = 0;
        std::string fetched_tags = std::string((char*)sqlite3_column_text(pStmt, 7));
        while ((pos = fetched_tags.find(",")) != std::string::npos)
        {
            fetched_kernel->tags.insert(fetched_tags.substr(0, pos));
            fetched_tags.erase(0, pos + 1);
        }
        if (fetched_tags != "")
        {
            fetched_kernel->tags.insert(fetched_tags);
        }

        // parse profiling information
        size_t subpos = 0;
        auto miscs = fetched_kernel->miscs;
        if (miscs.find("external_profile") != miscs.end())
        {
            std::string fetched_profile = miscs["external_profile"]["time"];
            while ((pos = fetched_profile.find(";")) != std::string::npos)
            {
                subpos = fetched_profile.find(":");
                fetched_kernel->profile[fetched_profile.substr(0, subpos)] =
                    stof(fetched_profile.substr(subpos + 1, pos));
                fetched_profile.erase(0, pos + 1);
            }

            fetched_kernel->resource = miscs["external_profile"]["resource"];
        }

        fetched.push_back(fetched_kernel);
    }

    NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
    if (fetched.size() > 0)
    {
        NNFUSION_LOG(DEBUG) << fetched.size() << " cached kernel fetched " << identifier
                            << " on: " << device_type;
    }
    else
    {
        NNFUSION_LOG(DEBUG) << "Failed to fetch, fallback plan will be uses";
    }
    return fetched;
}

KernelEntry_p KernelCacheManager::fetch_with_tags(std::string identifier,
                                                  std::string device_type,
                                                  std::set<std::string> tags,
                                                  bool efficient)
{
    auto fetched = fetch_all(identifier, device_type);

    std::string device = FLAGS_fproduct_name;
    KernelEntry target_kernel;
    target_kernel.profile[device] = 1048576;

    for (int i = 0; i < fetched.size(); i++)
    {
        for (auto tag : tags)
        {
            if (fetched[i]->tags.find(tag) == fetched[i]->tags.end())
            {
                fetched.erase(fetched.begin() + i--);
            }
        }
    }
    if (fetched.empty())
    {
        NNFUSION_LOG(DEBUG) << "No kernel with the same tags matched for " << identifier;
        return nullptr;
    }
    else
    {
        target_kernel.function = fetched[0]->function;
        target_kernel.resource = fetched[0]->resource;
        for (auto matched_kernel : fetched)
        {
            if (matched_kernel->profile.find(device) != matched_kernel->profile.end())
            {
                if ((efficient &&
                     matched_kernel->profile[device] * matched_kernel->resource <
                         target_kernel.profile[device] * target_kernel.resource) ||
                    ((!efficient) &&
                     matched_kernel->profile[device] < target_kernel.profile[device]))
                {
                    target_kernel.function = matched_kernel->function;
                    target_kernel.profile[device] = matched_kernel->profile[device];
                    target_kernel.resource = matched_kernel->resource;
                }
            }
        }
    }
    if (target_kernel.profile[device] == 1048576)
    {
        // set 10 as the default;
        target_kernel.profile[device] = 10;
    }
    return std::make_shared<KernelEntry>(target_kernel);
}
std::vector<KernelEntry_p> KernelCacheManager::fetch_with_source(std::string identifier,
                                                                 std::string device_type,
                                                                 std::string source)
{
    auto fetched = fetch_all(identifier, device_type);

    for (int i = 0; i < fetched.size(); i++)
    {
        if (fetched[i]->source != source)
        {
            fetched.erase(fetched.begin() + i--);
        }
    }
    if (fetched.empty())
    {
        NNFUSION_LOG(DEBUG) << "No kernel with the same source matched for " << identifier;
    }

    return fetched;
}
bool KernelCacheManager::insert_kernel_entry(const KernelEntry_p kernel_entry, bool overwrite)
{
    if (kernel_entry == nullptr)
    {
        NNFUSION_LOG(ERROR) << "kernel_entry is nullptr, unable to insert into kernel cache DB";
        return false;
    }

    if (SupportOpList.find(kernel_entry->op_type) == SupportOpList.end())
    {
        NNFUSION_LOG(DEBUG) << "Unsupported op_type: " << kernel_entry->op_type
                            << ", unable to insert into kernel cache DB";
        return false;
    }

    NNFUSION_LOG(DEBUG) << "Trying to insert kernel " << kernel_entry->identifier
                        << " on DeviceType: " << kernel_entry->device_type;

    std::string key = kernel_entry->key;
    std::string identifier = kernel_entry->identifier;
    std::string op_type = kernel_entry->op_type;
    std::string attributes = kernel_entry->attributes.dump();
    std::string source = kernel_entry->source;
    std::string device_type = kernel_entry->device_type;
    std::string function = kernel_entry->function.dump();
    std::string miscs = kernel_entry->miscs.dump();

    std::string tags = "";
    for (auto tag : kernel_entry->tags)
    {
        if (tags == "")
        {
            tags = tag;
        }
        else
        {
            tags += "," + tag;
        }
    }

    NNFUSION_CHECK(key != "" && identifier != "" && op_type != "" && source != "" &&
                   device_type != "" && function != "");

    if (overwrite)
    {
        NNFUSION_LOG(DEBUG) << "Allow overwriting kernel " << kernel_entry->identifier
                            << " in kernel cache DB";
        sqlite3_stmt* pStmt;
        const char* sql_delete = R"(
DELETE FROM KernelCache WHERE (Key = ?);
        )";
        NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(kernel_cache, sql_delete, -1, &pStmt, 0));
        sqlite3_bind_text(pStmt, 1, key.data(), key.size(), SQLITE_STATIC);
        NNFUSION_CHECK(SQLITE_DONE == sqlite3_step(pStmt));
        NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
    }

    sqlite3_stmt* pStmt;
    const char* sql_insert = R"(
INSERT INTO KernelCache (Key,Identifier,OpType,Attributes,Source,DeviceType,Function,Tags,Miscs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";
    NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(kernel_cache, sql_insert, -1, &pStmt, 0));
    sqlite3_bind_text(pStmt, 1, key.data(), key.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 2, identifier.data(), identifier.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 3, op_type.data(), op_type.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 4, attributes.data(), attributes.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 5, source.data(), source.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 6, device_type.data(), device_type.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 7, function.data(), function.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 8, tags.data(), tags.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 9, miscs.data(), miscs.size(), SQLITE_STATIC);
    NNFUSION_CHECK(SQLITE_DONE == sqlite3_step(pStmt));
    NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));

    return true;
}