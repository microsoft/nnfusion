//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2019, NNFusion Team

#include "fused.hpp"
#include <algorithm>
#include <regex>
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::op;

std::string retarget_expr_mediates(std::string expr,
                                   int& mediate_offset,
                                   std::string mediate_base = "mediate")
{
    std::regex_token_iterator<std::string::iterator> rend;
    std::regex expr_sep(";");
    std::regex io_sep("=");
    std::regex mdt_sep("mediate\\d*");

    std::vector<std::string> individual_exprs;
    std::string plan;
    std::regex_token_iterator<std::string::iterator> exprs_pos(
        expr.begin(), expr.end(), expr_sep, -1);
    decltype(expr_sep) end;
    for (; exprs_pos != rend; ++exprs_pos)
        individual_exprs.push_back(exprs_pos->str());
    if (individual_exprs[individual_exprs.size() - 1].find("##") != std::string::npos)
    {
        plan = individual_exprs[individual_exprs.size() - 1];
        individual_exprs.pop_back();
    }

    std::vector<std::pair<std::string, std::string>> mediates_cast;
    int max_offset = -1;
    for (auto& ind_expr : individual_exprs)
    {
        auto output_str = ind_expr.substr(0, ind_expr.find("="));
        std::smatch matches;
        bool ret = std::regex_search(output_str, matches, mdt_sep);
        if (!ret)
            continue;
        for (auto m : matches)
        {
            auto mediate_name = m.str();
            auto mediate_index = std::stoi(mediate_name.substr(mediate_base.size()));
            max_offset = std::max(max_offset, mediate_index);
            auto casted_name = mediate_base + std::to_string(mediate_index + mediate_offset);
            mediates_cast.push_back(std::make_pair(mediate_name, casted_name));
        }
    }
    mediate_offset += (max_offset + 1);
    std::reverse(mediates_cast.begin(), mediates_cast.end());

    std::string retrgt_expr(expr);
    for (auto& m_cast : mediates_cast)
    {
        std::regex reg_cast(m_cast.first);
        retrgt_expr = std::regex_replace(retrgt_expr, reg_cast, m_cast.second);
    }

    return retrgt_expr;
}

void Fused::register_ir2(std::vector<std::shared_ptr<graph::GNode>>& gnodes)
{
    // DEBUG: Preprint the IR list of all gnodes
    // NNFUSION_LOG(INFO) << "Fusion IR list";
    // for (auto& m_node : gnodes)
    // {
    //     auto& configs = get_op_configs();
    //     auto it = configs.find(m_node->get_op_ptr()->get_op_type());
    //     NNFUSION_CHECK(it->second.f_translate_v2) << m_node->get_op_type();
    //     NNFUSION_LOG(INFO) << it->second.f_translate_v2(m_node);
    // }

    std::unordered_map<std::shared_ptr<graph::GNode>, std::vector<std::string>> outputs_info_dict;
    std::unordered_map<std::shared_ptr<graph::GNode>, std::vector<std::string>> inputs_info_dict;

    int output_offset = 0;
    int mediate_offset = 0;
    int input_offset = 0;
    std::unordered_set<std::shared_ptr<graph::GNode>> gnodes_set(gnodes.begin(), gnodes.end());
    for (auto& m_node : gnodes)
    {
        // Step 1: Get the template expression of target node
        auto& configs = get_op_configs();
        auto it = configs.find(m_node->get_op_ptr()->get_op_type());
        NNFUSION_CHECK(it->second.f_translate_v2) << m_node->get_op_type();
        auto mediate_expr_template = it->second.f_translate_v2(m_node);

        // Step 2: Retarget the internal offset of mediates
        mediate_expr_template = retarget_expr_mediates(mediate_expr_template, mediate_offset);

        OpConfig::any ioConfig;

        // Step 3: Alignment the input dictory to the expression
        int node_input_offset = 0;
        std::vector<std::string> input_info(m_node->get_input_size());
        for (int in_id = 0; in_id < m_node->get_input_size(); ++in_id)
        {
            auto& in_edge = m_node->get_in_edge(in_id);
            if (in_edge == nullptr || in_edge->is_control_edge())
                continue;
            auto in_node = outputs_info_dict.find(in_edge->get_src());
            if (in_node == outputs_info_dict.end())
            {
                input_info[in_id] = "@input" + to_string(input_offset++) + "@";
            }
            else
            {
                input_info[in_id] = in_node->second[in_edge->get_src_output()];
            }
            ioConfig["input" + to_string(node_input_offset++)] = input_info[in_id];
        }
        inputs_info_dict.insert(std::make_pair(m_node, input_info));

        // Step 4: Alignment the output dictory to the expression
        std::vector<std::string> output_info(m_node->get_output_size());
        int node_output_offset = 0;
        std::unordered_set<int> output_ids;
        for (auto& out_edge : m_node->get_out_edges())
        {
            if (gnodes_set.find(out_edge->get_dst()) == gnodes_set.end())
            {
                auto out_id = out_edge->get_src_output();
                if (output_ids.find(out_id) == output_ids.end())
                {
                    output_info[out_edge->get_src_output()] =
                        "@output" + to_string(output_offset++) + "@";
                    output_ids.insert(out_id);
                }
            }
        }
        for (auto& out_edge : m_node->get_out_edges())
        {
            if (output_info[out_edge->get_src_output()].empty())
            {
                output_info[out_edge->get_src_output()] = "mediate" + to_string(mediate_offset++);
            }
        }
        for (auto& out_i : output_info)
        {
            ioConfig["output" + to_string(node_output_offset++)] = out_i;
        }
        outputs_info_dict.insert(std::make_pair(m_node, output_info));

        // Step 5: Create fused expression and plan
        std::string mediate_expr;
        mediate_expr = create_code_from_template(mediate_expr_template, ioConfig);
        auto rule_split = mediate_expr.find("## @: ");
        fused_op_ir2 = fused_op_ir2 + mediate_expr.substr(0, rule_split);
        if (rule_split != std::string::npos)
        {
            auto extra_plan = mediate_expr.substr(rule_split + 5);
            if (extra_plan.find_first_not_of(" ") != std::string::npos)
                plan_rules.push_back(extra_plan);
        }
    }

    // DEBUG: Print fused IR
    // NNFUSION_LOG(INFO) << fused_op_ir2;
}
std::string Fused::get_plan_rule()
{
    // plan_rule = "## @: " + plan_rule;
    std::string plan_expr = "## @: ";
    if (plan_rules.size() == 1)
    {
        return plan_expr + plan_rules[0];
    }
    else if (plan_rules.size() > 1)
    {
        // return plan_expr + "plan/multi_reduce";
        return plan_expr;
    }
    return plan_expr;
}
