// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include "generic_op.hpp"
namespace nnfusion
{
    namespace op
    {
        std::unordered_map<std::string, OpConfig>& get_op_configs()
        {
            static std::unordered_map<std::string, OpConfig> __op_configs;
            return __op_configs;
        }

        // empty string result when translation is not available for a certain op
        std::string get_translation(std::shared_ptr<nnfusion::graph::GNode>& gnode)
        {
            std::string result = get_translation_v2(gnode);
            if (!result.empty())
                return result;
            auto& configs = get_op_configs();
            auto it = configs.find(gnode->get_op_ptr()->get_op_type());
            if (it == configs.end() || it->second.f_translate == nullptr)
                return "";
            result = it->second.f_translate(gnode);
            return std::move(result);
        }

        std::string get_translation_v2(std::shared_ptr<nnfusion::graph::GNode>& gnode)
        {
            auto& configs = get_op_configs();
            auto it = configs.find(gnode->get_op_ptr()->get_op_type());
            auto antares_template = get_ir_via_extension(gnode);
            if (antares_template == "")
            {
                if (it == configs.end() || it->second.f_translate_v2 == nullptr)
                    return "";

                antares_template = it->second.f_translate_v2(gnode);
                if (antares_template.empty())
                    return "";
            }
            op::OpConfig::any io_config;
            auto input_template =
                R"( "@input@" : { "dtype" : "@input_dtype@", "shape" : @input_shape@} )";
            std::string input_code;
            int real_input_id = 0;
            for (int in_id = 0; in_id < gnode->get_input_size(); ++in_id)
            {
                auto& in_edge = gnode->get_in_edge(0);
                if (in_edge == nullptr || in_edge->is_control_edge())
                    continue;
                op::OpConfig::any input_config;
                auto input_alias = "input" + to_string(real_input_id++);
                op::create_inputs_definition_from_tensor(
                    gnode->get_input_tensor_ptr(in_id), input_alias, input_config, "input");
                if (antares_template.find(input_alias) != std::string::npos)
                {
                    input_code = input_code + (input_code.empty() ? "" : ", ") +
                                 op::create_code_from_template(input_template, input_config);
                }
                io_config[input_alias] = input_alias;
            }
            std::vector<std::string> extra_outputs;
            for (int out_id = 0; out_id < gnode->get_output_size(); ++out_id)
            {
                auto output_alias = "output" + to_string(out_id);
                io_config[output_alias] = output_alias;
                extra_outputs.push_back("\"" + output_alias + "\"");

                // special handle for "=." expr, which need add output into input dict
                if (antares_template.find("=.") != std::string::npos)
                {
                    op::OpConfig::any output_config;
                    op::create_inputs_definition_from_tensor(
                        gnode->get_output_tensor_ptr(out_id), output_alias, output_config, "input");
                    input_code = input_code + (input_code.empty() ? "" : ", ") +
                                 op::create_code_from_template(input_template, output_config);
                }
            }
            auto extra_output = join(extra_outputs, ", ");

            std::string expression_code =
                op::create_code_from_template(antares_template, io_config);

            int plan_pos = expression_code.find("## @");
            std::string plan =
                (std::string::npos == plan_pos) ? "" : expression_code.substr(plan_pos);
            expression_code = expression_code.substr(0, plan_pos);

            std::string ir_code;
            if (extra_outputs.size() > 1)
            {
                ir_code = op::create_code_from_template(
                    R"( - einstein_v2("@exp_code@", input_dict={@in_code@}, extra_outputs=[@extra_outputs@]) @plan@ )",
                    {{"exp_code", expression_code},
                     {"in_code", input_code},
                     {"extra_outputs", extra_output},
                     {"plan", plan}});
            }
            else
            {
                ir_code = op::create_code_from_template(
                    R"( - einstein_v2("@exp_code@", input_dict={@in_code@}) @plan@ )",
                    {{"exp_code", expression_code}, {"in_code", input_code}, {"plan", plan}});
            }
            return std::move(ir_code);
        }

        std::string get_annotation(std::string translation)
        {
            std::string options;
            const char annotation[] = "## @: ";
            int pos = translation.find(annotation);
            if (pos >= 0)
            {
                pos += sizeof(annotation) - 1;
                options = translation.substr(pos);
                options.erase(remove(options.begin(), options.end(), ' '), options.end());
                std::replace(options.begin(), options.end(), ',', '|');
            }

            if (options.size() > 0)
            {
                if (options[0] != '|')
                    options = "|" + options;
                if (options.back() != '|')
                    options += "|";
            }

            return options;
        }

        std::string get_ir_via_extension(std::shared_ptr<graph::GNode> gnode)
        {
            nnfusion::json message;
            message["output_name"] = "@output0@";
            std::vector<nnfusion::json> input_dict;
            for (size_t i = 0; i < gnode->get_input_size(); i++)
            {
                nnfusion::json input_info;
                input_info["name"] = "@input" + to_string(i) + "@";
                input_info["dtype"] = gnode->get_input_element_type(i).c_type_string();
                input_info["shape"] = gnode->get_input_shape(i);
                input_dict.push_back(input_info);
            }
            message["input_dict"] = input_dict;
            nnfusion::json config = gnode->get_op_ptr()->serialize();
            if (config.empty())
            {
                // NNFUSION_LOG(NNFUSION_WARNING) << "config for " << gnode->get_op_type()
                //                                << " is empty";
                return "";
            }
            message["config"] = config;

            std::string file_path = "./extensions/" + gnode->get_op_type();
            struct stat buffer;
            if (stat(file_path.c_str(), &buffer) != 0)
            {
                // NNFUSION_LOG(NNFUSION_WARNING) << "extension for " << gnode->get_op_type()
                //                                << " does not exist";
                return "";
            }

            std::string cmd = file_path + " '" + message.dump() + "'", ir_string;
            NNFUSION_LOG(INFO) << "Execute: " << cmd;

            static char line[4096];
            FILE* fp = popen(cmd.c_str(), "r");
            while (fgets(line, sizeof(line), fp))
                ir_string += line;
            pclose(fp);
            ir_string.pop_back(); // romove '\n'
            NNFUSION_LOG(INFO) << "Response: " << ir_string;
            return ir_string;
        }
        // +        std::string get_annotation(std::string translation)
        // +        {
        // +            std::string options;
        // +            const char annotation[] = "## @annotation: ";
        // +            int pos = translation.find(annotation);
        // +            if (pos >= 0)
        // +            {
        // +                pos += sizeof(annotation) - 1;
        // +                options = translation.substr(pos);
        // +            }
        // +
        // +            if (options.size() > 0)
        // +            {
        // +                if (options[0] != '|')
        // +                    options = "|" + options;
        // +                if (options.back() != '|')
        // +                    options += "|";
        // +            }
        // +
        // +            return options;
        // +        }
    } // namespace op
} // namespace nnfusion
