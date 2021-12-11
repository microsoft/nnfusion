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

#include "backward_registry.hpp"

using namespace nnfusion::pass::graph::autodiff;

nlohmann::json nnfusion::pass::graph::autodiff::training_optimizer_configs;

void DiffEngine::add_delta(const GNodeIndex& x, const GNodeIndex& grad)
{
    auto adjoint_it = m_adjoint_map.find(x.gnode);
    if (adjoint_it == m_adjoint_map.end())
    {
        m_adjoint_map[x.gnode] = GNodeIndexVector(x.gnode->get_output_size());
        adjoint_it = m_adjoint_map.find(x.gnode);
    }
    auto& deltas = adjoint_it->second[x.index];
    if (deltas == EMPTY_GNODE_INDEX)
    {
        deltas = grad;
    }
    else
    {
        auto accumulator = m_graph->add_node_and_edge(std::make_shared<op::Add>(), {deltas, grad});
        deltas = GNodeIndex{accumulator, 0};
    }
}

void DiffEngine::differentiate_graph(const GNodeIndexVector& outputs,
                                     const GNodeIndexVector& outputs_grad)
{
    NNFUSION_CHECK(outputs.size() == outputs_grad.size())
        << "outputs and outputs_grad must be equal size";

    // Pass 1 determines which nodes contribute to y as well as setting up a reverse
    // topological sort.

    // Number of nodes that use the node's value
    std::unordered_map<std::shared_ptr<GNode>, size_t> parent_counts;

    // Nodes we should check
    std::list<std::shared_ptr<GNode>> nodes_to_check;
    for (auto& output : outputs)
    {
        nodes_to_check.push_back(output.gnode);
    }
    auto& backward_registry = Registry();
    std::unordered_set<std::string> no_backward_ops;
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        if (backward_registry.find(node->get_op_type()) == backward_registry.end())
        {
            no_backward_ops.insert(node->get_op_type());
        }
        nodes_to_check.pop_front();
        if (m_adjoint_map.find(node) == m_adjoint_map.end())
        {
            m_adjoint_map[node] = GNodeIndexVector(node->get_output_size());
            for (auto in_edge : node->get_in_edges())
            {
                auto in_gnode = in_edge->get_src();
                auto count_it = parent_counts.find(in_gnode);
                if (count_it == parent_counts.end())
                {
                    parent_counts[in_gnode] = 1;
                    nodes_to_check.push_front(in_gnode);
                }
                else
                {
                    parent_counts[in_gnode]++;
                }
            }
        }
    }
    if (no_backward_ops.size() > 0)
    {
        for (auto& op_type : no_backward_ops)
        {
            NNFUSION_LOG(ERROR) << "No backward translator for op " << op_type;
        }
        NNFUSION_CHECK_FAIL();
    }

    // Second pass visits the nodes so that all users of a node's value are visited
    // before a node is visited.
    for (size_t i = 0; i < outputs.size(); i++)
    {
        add_delta(outputs.at(i), outputs_grad.at(i));
    }

    for (auto& output : outputs)
    {
        auto node = output.gnode;
        if (find(nodes_to_check.begin(), nodes_to_check.end(), node) == nodes_to_check.end())
        {
            nodes_to_check.push_back(node);
        }
    }

    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        // Look for nodes that will be available when this node is done
        // sort in_edges by output index
        auto in_edges = node->get_in_edges();
        std::vector<std::shared_ptr<Edge>> sorted_in_edges(in_edges.begin(), in_edges.end());
        std::sort(sorted_in_edges.begin(),
                  sorted_in_edges.end(),
                  [](std::shared_ptr<Edge> a, std::shared_ptr<Edge> b) {
                      return a->get_dst_input() < b->get_dst_input();
                  });
        for (auto in_edge : sorted_in_edges)
        {
            auto input_source_node = in_edge->get_src();
            auto count_it = parent_counts.find(input_source_node);
            count_it->second--;
            if (0 == count_it->second)
            {
                nodes_to_check.push_back(input_source_node);
            }
        }
        GNodeIndexVector deltas = m_adjoint_map[node];
        // fill empty grads
        for (size_t i = 0; i < node->get_output_size(); i++)
        {
            auto& delta = deltas[i];
            if (delta == EMPTY_GNODE_INDEX)
            {
                auto zero_op = std::make_shared<op::Constant>(
                    element::f32, node->get_output_shape(i), std::vector<float>{0});
                zero_op->set_name(node->get_name() + "_grad_" + std::to_string(i));
                auto zero = m_graph->add_node_and_edge(zero_op, GNodeVector());
                delta = GNodeIndex{zero, 0};
            }
        }

        // backward from node output to input
        auto& backward_registry = Registry();
        auto it = backward_registry.find(node->get_op_type());
        NNFUSION_CHECK(it != backward_registry.end()) << "backward translator not found for "
                                                      << node->get_op_type();
        GNodeIndexVector inputs_grad = it->second.m_translator(node, deltas, m_graph);
        NNFUSION_CHECK(inputs_grad.size() == node->get_input_size())
            << "inputs and inputs_grad must be equal size";
        for (auto& in_edge : node->get_in_edges())
        {
            auto dst_index = in_edge->get_dst_input();
            if (inputs_grad[dst_index] != EMPTY_GNODE_INDEX)
            {
                add_delta(GNodeIndex(in_edge->get_src(), in_edge->get_src_output()),
                          inputs_grad[dst_index]);
            }
        }
        ///\todo how to optimize? option: for parameter node, its backward Translator is an optimizer
    }
}

const GNodeIndex DiffEngine::EMPTY_GNODE_INDEX;