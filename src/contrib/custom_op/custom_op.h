// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <dirent.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
using json = nlohmann::json;

#define REGISTER_CUSTOM_OP(op_x) REGISTER_OP(op_x)

class CustomOpsRegistration
{
public:
    CustomOpsRegistration(std::string type)
        : base_dir("./custom_op/" + type)
    {
        DIR* dirp = opendir(base_dir.c_str());
        struct dirent* dp = readdir(dirp);

        while (dp != NULL)
        {
            std::string file(dp->d_name);
            if (file.length() > 4)
            {
                std::string data_path = base_dir + "/" + file;
                std::cout << data_path << std::endl;
                if (type == "json")
                {
                    register_json_ops(data_path);
                }
                else if (type == "onnx")
                {
                    register_onnx_ops(data_path);
                }
                else if (type == "tensorflow")
                {
                    register_tf_ops(data_path);
                }
                else
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "unrecongnized op type!";
                }
                
            }
            dp = readdir(dirp);
        }
        closedir(dirp);
    }

    bool register_json_ops(std::string data_path);
    bool register_onnx_ops(std::string data_path) {return false;}
    bool register_tf_ops(std::string data_path) {return false;}
    void register_common(nnfusion::op::OpConfig& op_reg);

    std::string base_dir;
};