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
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::op;

void Fused::register_ir2(std::vector<std::shared_ptr<graph::GNode>>& gnodes)
{
    std::unordered_map<std::shared_ptr<graph::GNode>, std::vector<std::string>> outputs_info_dict;
    std::unordered_map<std::shared_ptr<graph::GNode>, std::vector<std::string>> inputs_info_dict;

    int output_offset = 0;
    int mediate_offset = 0;
    int input_offset = 0;
    std::unordered_set<std::shared_ptr<graph::GNode>> gnodes_set(gnodes.begin(), gnodes.end());
    for (auto& m_node : gnodes)
    {
        int node_input_offset = 0;
        int node_output_offset = 0;
        OpConfig::any ioConfig;

        auto num_inputs = 0;
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

        std::vector<std::string> output_info(m_node->get_output_size());
        for (auto& out_edge : m_node->get_out_edges())
        {
            if (gnodes_set.find(out_edge->get_dst()) == gnodes_set.end())
            {
                output_info[out_edge->get_src_output()] =
                    "@output" + to_string(output_offset++) + "@";
            }
            else if (output_info[out_edge->get_src_output()].empty())
            {
                output_info[out_edge->get_src_output()] = "mediate" + to_string(mediate_offset++);
            }
            ioConfig["output" + to_string(node_output_offset++)] =
                output_info[out_edge->get_src_output()];
        }
        outputs_info_dict.insert(std::make_pair(m_node, output_info));

        auto& configs = get_op_configs();
        auto it = configs.find(m_node->get_op_ptr()->get_op_type());
        auto mediate_ir2 = create_code_from_template(it->second.f_translate_v2(m_node), ioConfig);
        auto rule_split = mediate_ir2.find("## @: ");
        fused_op_ir2 = fused_op_ir2 + mediate_ir2.substr(0, rule_split);
        if (rule_split != std::string::npos)
        {
            plan_rule = plan_rule + mediate_ir2.substr(rule_split + 5);
        }
    }

    plan_rule = "## @: " + plan_rule;
}
