// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <dirent.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdlib.h>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
using json = nlohmann::json;

#define REGISTER_CUSTOM_OP(op_x) REGISTER_OP(op_x)

class CustomOpsRegistration
{
public:
    CustomOpsRegistration(std::string type)
    {
        char* nnfusion_home = getenv("NNFUSION_HOME");
        if (nnfusion_home == NULL)
        {
            char* home = getenv("HOME");
            if (home != NULL)
            {
                base_dir = std::string(home) + "/nnfusion/custom_op/" + type;
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "$NNFUSION_HOME was not set, use " << std::string(home) << "/nnfusion.";
            }
        }
        else
        {
            base_dir = std::string(nnfusion_home) + "/custom_op/" + type;
        }

        if (base_dir != "")
        {
            DIR* dirp = opendir(base_dir.c_str());
            if (dirp != NULL)
            {
                struct dirent* dp = readdir(dirp);

                while (dp != NULL)
                {
                    std::string file(dp->d_name);
                    if (file.length() > 4)
                    {
                        std::string data_path = base_dir + "/" + file;
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
        }
    }

    bool register_json_ops(std::string data_path);
    bool register_onnx_ops(std::string data_path) { return false; }
    bool register_tf_ops(std::string data_path) { return false; }
    void register_common(nnfusion::op::OpConfig& op_reg);

    std::string base_dir;
};