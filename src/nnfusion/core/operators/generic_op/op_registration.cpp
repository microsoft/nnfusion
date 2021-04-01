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
            if (it == configs.end() || it->second.f_translate_v2 == nullptr)
                return "";

            auto antares_template = it->second.f_translate_v2(gnode);
            if (antares_template.empty())
                return "";

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
            for (int out_id = 0; out_id < gnode->get_output_size(); ++out_id)
            {
                auto output_alias = "output" + to_string(out_id);
                io_config[output_alias] = output_alias;
            }

            std::string expression_code =
                op::create_code_from_template(antares_template, io_config);

            int plan_pos = expression_code.find("## @");
            std::string plan =
                (std::string::npos == plan_pos) ? "" : expression_code.substr(plan_pos);
            expression_code = expression_code.substr(0, plan_pos);

            std::string ir_code = op::create_code_from_template(
                R"( - einstein_v2("@exp_code@", input_dict={@in_code@}) @plan@ )",
                {{"exp_code", expression_code}, {"in_code", input_code}, {"plan", plan}});
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
