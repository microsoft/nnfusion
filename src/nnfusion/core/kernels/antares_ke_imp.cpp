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

        NNFUSION_LOG(INFO) << "[Autogen] " << expr << " (tuned = " << tuned << ")";
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

std::vector<nnfusion::Shape> AntaresKEImp::get_output_shapes(const std::string& response)
{
    /*///1-128-8-8/float32/input0,3-3-128-1/float32/input1:1-128-8-8/float32/output0*/
    std::vector<nnfusion::Shape> output_shapes;

    size_t pos_st = response.find("///");
    NNFUSION_CHECK(pos_st != std::string::npos);
    pos_st += 3; // size of "///"
    size_t pos_ed = response.find("\n", pos_st);
    NNFUSION_CHECK(pos_ed != std::string::npos && pos_ed > pos_st);
    std::string shape_string = response.substr(pos_st, pos_ed - pos_st);
    NNFUSION_CHECK(shape_string.find("output") != std::string::npos);

    pos_st = shape_string.find_last_of(":");
    NNFUSION_CHECK(pos_st != std::string::npos);
    pos_st += 1;
    std::string output_shape_string = shape_string.substr(pos_st, shape_string.size() - pos_st);
    NNFUSION_CHECK(output_shape_string.find(":") == std::string::npos);

    std::vector<std::string> outputs;
    output_shape_string.push_back(',');
    {
        size_t curr, prev = 0;
        curr = output_shape_string.find(",");
        while (curr != std::string::npos)
        {
            outputs.push_back(output_shape_string.substr(prev, curr - prev));
            prev = curr + 1;
            curr = output_shape_string.find(',', prev);
        }
    }
    for (auto output : outputs)
    {
        std::string shape_str = output.substr(0, output.find('/'));
        nnfusion::Shape shape;

        shape_str.push_back('-');
        size_t curr, prev = 0;
        curr = shape_str.find("-");
        while (curr != std::string::npos)
        {
            shape.push_back(std::stoi(shape_str.substr(prev, curr - prev)));
            prev = curr + 1;
            curr = shape_str.find("-", prev);
        }
        output_shapes.push_back(shape);
    }

    return output_shapes;
}

std::vector<AntaresKernelInfo::Pointer> AntaresKEImp::get_kernel_info(const std::string& response)
{
    // example:
    // LOCAL: template_op_kernel0 -- input0:float32[1, 12, 576, 576] -> mediate0:float32[1, 12, 576]
    // ...
    // LOCAL: template_op_kernel1 -- input0:float32[1, 12, 576, 576], mediate0:float32[1, 12, 576] -> mediate1:float32[1, 12, 576]
    // ...
    // LOCAL: template_op_kernel2 -- input0:float32[1, 12, 576, 576], mediate0:float32[1, 12, 576], mediate1:float32[1, 12, 576] -> output0:float32[1, 12, 576, 576]

    std::vector<AntaresKernelInfo::Pointer> res;

    std::string start = "// LOCAL: ";
    std::string end = "\n";
    std::vector<std::string> kernel_info_cmt;

    int s_pos = response.find(start, 0);

    while (s_pos > 0)
    {
        int e_pos = response.find(end, s_pos);
        NNFUSION_CHECK(e_pos >= 0);
        std::string info = response.substr(s_pos + 10, e_pos - s_pos - 10);
        kernel_info_cmt.push_back(info);
        s_pos = response.find(start, e_pos);
    }

    for (size_t i = 0; i < kernel_info_cmt.size(); i++)
    {
        std::string info = kernel_info_cmt[i];
        AntaresKernelInfo::Pointer kernel_info = std::make_shared<AntaresKernelInfo>();

        int kernel_name_s_pos = 0;
        int kernel_name_e_pos = info.find(" -- ", kernel_name_s_pos);
        NNFUSION_CHECK(kernel_name_e_pos >= 0);
        kernel_info->kernel_name =
            info.substr(kernel_name_s_pos, kernel_name_e_pos - kernel_name_s_pos);
        int input_name_e_pos = info.find(" -> ", kernel_name_e_pos);
        NNFUSION_CHECK(input_name_e_pos >= 0);
        std::string input_info =
            info.substr(kernel_name_e_pos + 4, input_name_e_pos - kernel_name_e_pos - 4);
        std::string output_info = info.substr(input_name_e_pos + 4);

        int single_input_s_pos = 0;
        int single_input_e_pos = input_info.find("]", single_input_s_pos);
        while (single_input_e_pos > 0)
        {
            std::string input =
                input_info.substr(single_input_s_pos, single_input_e_pos - single_input_s_pos + 1);
            int name_e_pos = input.find(":");
            NNFUSION_CHECK(name_e_pos >= 0);
            int shape_s_pos = input.find("[");
            NNFUSION_CHECK(shape_s_pos >= 0);
            kernel_info->input_names.push_back(input.substr(0, name_e_pos));
            kernel_info->input_dtypes.push_back(
                input.substr(name_e_pos + 1, shape_s_pos - name_e_pos - 1));
            kernel_info->input_shapes.push_back(input.substr(shape_s_pos));

            single_input_s_pos = single_input_e_pos + 3;
            single_input_e_pos = input_info.find("]", single_input_s_pos);
        }

        int single_output_s_pos = 0;
        int single_output_e_pos = output_info.find("]", single_output_s_pos);
        while (single_output_e_pos > 0)
        {
            std::string output = output_info.substr(single_output_s_pos,
                                                    single_output_e_pos - single_output_s_pos + 1);
            int name_e_pos = output.find(":");
            NNFUSION_CHECK(name_e_pos >= 0);
            int shape_s_pos = output.find("[");
            NNFUSION_CHECK(shape_s_pos >= 0);
            kernel_info->output_names.push_back(output.substr(0, name_e_pos));
            kernel_info->output_dtypes.push_back(
                output.substr(name_e_pos + 1, shape_s_pos - name_e_pos - 1));
            kernel_info->output_shapes.push_back(output.substr(shape_s_pos));

            single_output_s_pos = single_output_e_pos + 3;
            single_output_e_pos = output_info.find("]", single_output_s_pos);
        }
        res.push_back(kernel_info);
    }

    return res;
}