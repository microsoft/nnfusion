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

double AntaresKEImp::get_perf(const std::string& response)
{
    /*// Saved Perf = 2.211810e-06 sec / run; Step Produced = 10; Planned Steps = 10;*/
    double perf = -1;
    std::string identifier_st = "\n// Saved Perf = ";
    std::string identifier_ed = " sec / run";
    size_t pos = response.find(identifier_st);
    bool tuned = pos != std::string::npos;
    if (tuned)
    {
        pos += identifier_st.size() - 1;
        size_t pos_ed = response.find(identifier_ed, pos);
        perf = std::stod(response.substr(pos, pos_ed - pos));
    }
    return perf;
}

std::pair<int, int> AntaresKEImp::get_tuning_step(const std::string& response)
{
    /*// Saved Perf = 2.211810e-06 sec / run; Step Produced = 10; Planned Steps = 10;*/
    int step_produced = -1;
    int planned_steps = -1;

    size_t pos = response.find("\n// Saved Perf = ");
    bool tuned = pos != std::string::npos;
    if (tuned)
    {
        size_t pos_st, pos_ed;

        std::string identifier_step_procuded = "Step Produced = ";
        pos_st = response.find(identifier_step_procuded, pos) + identifier_step_procuded.size() - 1;
        pos_ed = response.find(";", pos_st);
        NNFUSION_CHECK(pos_st != std::string::npos && pos_ed != std::string::npos);
        step_produced = std::stoi(response.substr(pos_st, pos_ed - pos_st));

        std::string identifier_planned_steps = "Planned Steps = ";
        pos_st = response.find(identifier_planned_steps, pos) + identifier_planned_steps.size() - 1;
        pos_ed = response.find(";", pos_st);
        NNFUSION_CHECK(pos_st != std::string::npos && pos_ed != std::string::npos);
        planned_steps = std::stoi(response.substr(pos_st, pos_ed - pos_st));
    }

    return std::make_pair(step_produced, planned_steps);
}

std::string AntaresKEImp::get_device_name(const std::string& response)
{
    /*// BACKEND = c-cuda (Tesla V100-PCIE-16GB)*/
    /*// BACKEND = c-cuda (default)*/
    std::string device_name = "default";
    size_t pos = response.find("\n// BACKEND = ");
    if (pos != std::string::npos)
    {
        size_t pos_st = response.find("(", pos) + 1;
        size_t pos_ed = response.find(")", pos);
        NNFUSION_CHECK(pos_st != std::string::npos && pos_ed != std::string::npos);
        device_name = response.substr(pos_st, pos_ed - pos_st);
    }
    return device_name;
}