// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "custom_op.h"

bool CustomOpsRegistration::register_json_ops(std::string data_path)
{
    std::ifstream fin(data_path);
    json op_config;
    fin >> op_config;
    fin.close();
    for (auto op : op_config["ops"])
    {
        std::string op_type = op["op"];
        nnfusion::op::OpConfig& op_reg = nnfusion::op::build_op_config(op_type);
        for (auto attr : op.items())
        {
            op_reg.getRoot()[attr.key()] = attr.value();
        }
        register_common(op_reg);
    }
    return true;
}

void CustomOpsRegistration::register_common(nnfusion::op::OpConfig& op_reg)
{
    // infer shapes
    op_reg.infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto& shape_0 = gnode->get_input_shape(0);

        auto op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
        auto inshapes = op_reg.getRoot()["input_shapes"];
        for (size_t i = 0; i < inshapes.size(); i++)
        {
            nnfusion::Shape in_shape_t;
            for (auto d : inshapes[i])
                in_shape_t.push_back(d);
            if (in_shape_t == shape_0)
            {
                auto outshapes = op_reg.getRoot()["output_shapes"];
                NNFUSION_CHECK(i < outshapes.size());
                nnfusion::Shape out_shape_t;
                for (auto d : outshapes[i])
                    out_shape_t.push_back(d);
                gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), out_shape_t);
                return;
            }
        }
        // by default set up the same as input shape
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    });

    // kernel functions
    if (op_reg.getRoot().find("antares_ir") != op_reg.getRoot().end())
    {
        op_reg.antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
            auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
            auto ir = op_config.get("antares_ir");
            return op::create_code_from_template(ir, op_config.getRoot());
        });
    }

    if (op_reg.getRoot().find("cpu_kernel") != op_reg.getRoot().end())
    {
        op_reg.cpu_kernel([](std::shared_ptr<graph::GNode> gnode) -> std::string {
            auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
            auto kernel = op_config.get("cpu_kernel");
            return op::create_code_from_template(kernel, op_config.getRoot());
        });
    }

    if (op_reg.getRoot().find("cuda_kernel") != op_reg.getRoot().end())
    {
        auto lconf_j = op_reg.getRoot()["launch_config"];
        NNFUSION_CHECK(lconf_j.size() == 2);
        std::vector<uint32_t> launch_config;
        for (size_t i = 0; i < 2; i++)
        {
            NNFUSION_CHECK(lconf_j[i].size() == 3);
            for (size_t j = 0; j < 3; j++)
            {
                launch_config.push_back(lconf_j[i][j]);
            }
        }
        bool is_memcpy = false;
        if (op_reg.getRoot().find("is_memcpy") != op_reg.getRoot().end())
        {
            is_memcpy = op_reg.getRoot()["is_memcpy"];
        }

        op_reg.cuda_kernel(
            [](std::shared_ptr<graph::GNode> gnode) -> std::string {
                auto op_config = nnfusion::op::lookup_op_config(gnode->get_op_type());
                auto kernel = op_config.get("cuda_kernel");
                NNFUSION_LOG(INFO) << kernel;
                auto r = op::create_code_from_template(kernel, op_config.getRoot());
                NNFUSION_LOG(INFO) << r;
                return r;
            },
            launch_config,
            is_memcpy);
    }
}

static CustomOpsRegistration jason_registra("json");
// static CustomOpsRegistration onnx_registra("onnx");
// static CustomOpsRegistration tensorflow_registra("tensorflow");