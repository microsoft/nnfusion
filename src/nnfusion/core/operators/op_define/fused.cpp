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
        std::regex reg_cast(m_cast.first + "\\[");
        retrgt_expr = std::regex_replace(retrgt_expr, reg_cast, m_cast.second + "[");
    }

    return retrgt_expr;
}

void Fused::register_ir2(std::vector<std::shared_ptr<graph::GNode>>& gnodes,
                         std::shared_ptr<graph::GNode> fused_node)
{
    std::string names;
    for (auto n : gnodes)
    {
        names += n->get_name() + ",";
    }
    fused_node->set_member_name(names);
    // DEBUG: Preprint the IR list of all gnodes
    // NNFUSION_LOG(INFO) << "Fusion IR list";
    // for (auto& m_node : gnodes)
    // {
    //     auto& configs = get_op_configs();
    //     auto it = configs.find(m_node->get_op_ptr()->get_op_type());
    //     NNFUSION_CHECK(it->second.f_translate_v2) << m_node->get_op_type();
    //     NNFUSION_LOG(INFO) << it->second.f_translate_v2(m_node);
    // }
    using index_dict_t =
        std::unordered_map<std::shared_ptr<graph::GNode>, std::unordered_map<int, std::string>>;
    index_dict_t inputs_dict;
    index_dict_t outputs_dict;
    index_dict_t mediate_dict;

    for (auto in_edge : fused_node->get_in_edges())
    {
        if (in_edge->is_control_edge())
            continue;
        inputs_dict[in_edge->get_src()][in_edge->get_src_output()] =
            "@input" + to_string(in_edge->get_dst_input()) + "@";
    }
    for (auto out_edge : fused_node->get_out_edges())
    {
        if (out_edge->is_control_edge())
            continue;
        outputs_dict[out_edge->get_dst()][out_edge->get_dst_input()] =
            "@output" + to_string(out_edge->get_src_output()) + "@";
    }

    int mediate_offset = 0;
    is_memcpy = true;
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
        for (auto in_edge : m_node->get_in_edges())
        {
            if (in_edge->is_control_edge())
                continue;
            auto iter = inputs_dict.find(in_edge->get_src());
            std::string input_str;
            if (iter != inputs_dict.end())
            {
                NNFUSION_CHECK(iter->second.find(in_edge->get_src_output()) != iter->second.end());
                input_str = iter->second[in_edge->get_src_output()];
            }
            else
            {
                auto med_iter = mediate_dict.find(in_edge->get_src());
                NNFUSION_CHECK(med_iter != mediate_dict.end());
                NNFUSION_CHECK(med_iter->second.find(in_edge->get_src_output()) !=
                               med_iter->second.end());
                input_str = med_iter->second[in_edge->get_src_output()];
            }
            ioConfig["input" + to_string(in_edge->get_dst_input())] = input_str;
        }

        // Step 4: Alignment the output dictory to the expression
        for (auto i = 0; i < m_node->get_output_size(); i++)
        {
            bool has_global_output = false;
            std::string output_str;
            auto out_edges = m_node->get_output_users(i);
            for (auto out_edge : out_edges)
            {
                auto iter = outputs_dict.find(out_edge->get_dst());
                if (iter != outputs_dict.end())
                {
                    NNFUSION_CHECK(iter->second.find(out_edge->get_dst_input()) !=
                                   iter->second.end());
                    output_str = iter->second[out_edge->get_dst_input()];
                    // update mediate_dict in case this slot has multiple users
                    mediate_dict[out_edge->get_src()][out_edge->get_src_output()] = output_str;
                    break;
                }
            }
            if (output_str.empty())
            {
                output_str = "mediate" + to_string(mediate_offset++);
                mediate_dict[m_node][i] = output_str;
            }
            ioConfig["output" + to_string(i)] = output_str;
        }

        // Step 5: Create fused expression and plan
        std::string mediate_expr;
        mediate_expr = create_code_from_template(mediate_expr_template, ioConfig);
        auto rule_split = mediate_expr.find("## @: ");
        fused_op_ir2 = fused_op_ir2 + mediate_expr.substr(0, rule_split);
        if (rule_split != std::string::npos)
        {
            auto extra_plan = mediate_expr.substr(rule_split + 5);
            if (extra_plan.find_first_not_of(" ") != std::string::npos)
            {
                plan_rules.push_back(extra_plan);
            }
            if (extra_plan.find("memcpy") == std::string::npos)
                is_memcpy = false;
        }
        else
        {
            is_memcpy = false;
        }
    }

    // DEBUG: Print fused IR
    // NNFUSION_LOG(INFO) << fused_op_ir2;
}
std::string Fused::get_plan_rule()
{
    // plan_rule = "## @: " + plan_rule;
    if (!is_memcpy)
    {
        std::vector<string> new_rules;
        // remove memcpy in each rule
        for (auto rule : plan_rules)
        {
            if (auto pos = rule.find("memcpy") != std::string::npos)
            {
                rule.erase(pos, 6);
                if (rule.find_first_not_of(" ") != std::string::npos)
                    new_rules.push_back(rule);
            }
            else
                new_rules.push_back(rule);
        }
        plan_rules = new_rules;
    }

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
