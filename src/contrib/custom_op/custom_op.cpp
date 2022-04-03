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

std::string exec(const char* cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

std::string replace_all(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

std::string get_base_dir()
{
    char* nnfusion_home = getenv("NNFUSION_HOME");
    std::string base_dir;
    if (nnfusion_home == NULL)
    {
        char* home = getenv("HOME");
        if (home != NULL)
        {
            base_dir = std::string(home) + "/nnfusion/";
            NNFUSION_LOG(NNFUSION_WARNING) << "$NNFUSION_HOME was not set, use "
                                           << std::string(home) << "/nnfusion.";
        }
    }
    else
    {
        base_dir = std::string(nnfusion_home);
    }

    return base_dir;
}

nlohmann::json execute_script(std::shared_ptr<graph::GNode> gnode)
{
    auto& op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
    auto& jsonroot = op_reg.getRoot();
    if (jsonroot.contains("script"))
    {
        using namespace nlohmann;
        json json_input = json::parse("{\"input\": {\"shape\": [], \"dtype\": []}}");
        // make json input
        auto script = jsonroot["script"].get<std::string>();
        auto op_name = jsonroot["op"].get<std::string>();
        auto base_dir = get_base_dir();

        for (int i = 0; i < gnode->get_input_size(); i++)
        {
            auto shape = gnode->get_inputs()[i]->get_shape();
            auto type = gnode->get_inputs()[i]->get_element_type();

            json_input["input"]["shape"].emplace_back(shape);
            json_input["input"]["dtype"].emplace_back(type.c_type_string());
        }

        // Put all constant node value into consideration
        //auto const_op = std::dynamic_pointer_cast<Constant>(gnode->get_inputs()[i]->get_op_ptr());
        //FIXME: Edge nubmer might not matching Input number
        auto const_data = map<int, vector<string>>();

        for(int i = 0; i < gnode->get_in_edges().size(); i++)
        {
            auto e = gnode->get_in_edge(i);
            auto n = e->get_src()->get_op_ptr();
            auto const_op = std::dynamic_pointer_cast<op::Constant>(n);
            if(const_op != nullptr)
            {
                auto strs = const_op->get_value_strings();
                const_data.insert(make_pair(i, strs));
            }
        }
        if(!const_data.empty())
            json_input["input"]["data"] = map<int, vector<string>>();

        for (auto attr : jsonroot.items())
        {
            json_input[attr.key()] = attr.value();
        }

        replace_all(script, "<NNFUSION_HOME>", base_dir);
        replace_all(script, "<OP_NAME>", op_name);
        std::string jstr = json_input.dump();
        json newjson = jstr;
        jstr = newjson.dump();
        replace_all(script, "<OP_JSON>", jstr);
        auto json_res = exec(script.c_str());
        auto json_out = json::parse(json_res);
        // NNFUSION_LOG(INFO) << json_res;

        return json_out;
    }
    return nlohmann::json("");
}

void CustomOpsRegistration::register_common(nnfusion::op::OpConfig& op_reg)
{
    // infer shapes
    op_reg.infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto& op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
        auto& jsonroot = op_reg.getRoot();
        if (jsonroot.contains("script"))
        {
            auto json_out = execute_script(gnode);
            if (json_out.contains("output"))
            {
                if (json_out["output"].contains("shape") && json_out["output"].contains("dtype"))
                {
                    for (int i = 0; i < json_out["output"]["shape"].size(); i++)
                    {
                        element::Type t;
                        std::string tn = json_out["output"]["dtype"][i].get<std::string>();
                        element::Type::dtype_string_to_nnfusion_element_type(tn, t);
                        auto d = json_out["output"]["shape"][i].get<std::vector<size_t>>();
                        nnfusion::Shape s(d.begin(), d.end());

                        gnode->set_output_type_and_shape(i, t, s);
                    }
                }
            }
            if (json_out.contains("antares_ir"))
            {
                op_reg.antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
                    auto& op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
                    auto out = execute_script(gnode);
                    auto ir = out["antares_ir"];
                    return op::create_code_from_template(ir, op_reg.getRoot());
                });
            }
            if (json_out.contains("cuda_kernel"))
            {
                op_reg.antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
                    auto& op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
                    auto out = execute_script(gnode);
                    auto ir = out["cuda_kernel"];
                    return op::create_code_from_template(ir, op_reg.getRoot());
                });
            }
            if (json_out.contains("hlsl_kernel"))
            {
                op_reg.antares_ir([](std::shared_ptr<graph::GNode> gnode) -> std::string {
                    auto& op_reg = nnfusion::op::lookup_op_config(gnode->get_op_type());
                    auto out = execute_script(gnode);
                    auto ir = out["hlsl_kernel"];
                    return op::create_code_from_template(ir, op_reg.getRoot());
                });
            }
        }
        else
        {
            auto& shape_0 = gnode->get_input_shape(0);
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
                    gnode->set_output_type_and_shape(
                        0, gnode->get_input_element_type(0), out_shape_t);
                    return;
                }
            }
            // by default set up the same as input shape
            gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
        }
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

static CustomOpsRegistration json_register("json");
static CustomOpsRegistration script_register("script");
// static CustomOpsRegistration onnx_registra("onnx");
// static CustomOpsRegistration tensorflow_registra("tensorflow");