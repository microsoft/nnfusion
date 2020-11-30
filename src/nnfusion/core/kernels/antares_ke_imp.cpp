// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "antares_ke_imp.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_string(fantares_codegen_server);

std::unordered_map<std::string, std::pair<std::string, bool>> AntaresKEImp::code_cache;

std::pair<std::string, bool> AntaresKEImp::autogen(const std::string& expr)
{
    if (FLAGS_fantares_codegen_server == "")
        return std::make_pair("", false); // FLAGS_fantares_codegen_server = "10.150.145.98:8881";

    std::string response;
    bool tuned = false;
    auto it = code_cache.find(expr);
    if (it == code_cache.end())
    {
        CurlRequest req(FLAGS_fantares_codegen_server);
        req.add_custom_header(("COMPUTE_V1: " + expr).c_str());

        if (!req.send_request(response))
        {
            NNFUSION_LOG(INFO) << "[Autogen] " << expr << " (tuned = " << -1 << ")";
            code_cache[expr] = std::make_pair(response, -1);
            return std::make_pair("", tuned);
        }
        if (strncmp(response.c_str(), "[ERROR]", 7) == 0)
        {
            NNFUSION_LOG(ERROR) << expr << "\n" << response;
            return std::make_pair("", tuned);
        }
        tuned = response.find("\n// Saved Perf =") != std::string::npos;

        // NNFUSION_LOG(INFO) << "[Autogen] " << expr << " (tuned = " << tuned << ")";
        code_cache[expr] = std::make_pair(response, tuned);
        return std::make_pair(std::move(response), tuned);
    }
    else
        return it->second;
}
