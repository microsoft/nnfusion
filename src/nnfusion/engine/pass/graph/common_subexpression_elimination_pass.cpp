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

// Microsoft (c) 2020, NNFusion Team

#include <climits>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common_subexpression_elimination_pass.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DEFINE_bool(fcse, false, "Common subexpression elimination.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

size_t hash_combine(const std::vector<size_t>& list)
{
    size_t seed = 0;
    for (size_t v : list)
    {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

static bool cse_broadcast(shared_ptr<GNode> a, shared_ptr<GNode> b)
{
    const auto broadcast_a = std::static_pointer_cast<op::Broadcast>(a->get_op_ptr());
    const auto broadcast_b = std::static_pointer_cast<op::Broadcast>(b->get_op_ptr());
    if (broadcast_a == nullptr || broadcast_b == nullptr)
    {
        return false;
    }
    return (broadcast_a->get_broadcast_axes() == broadcast_b->get_broadcast_axes()) &&
           (broadcast_a->get_broadcast_shape() == broadcast_b->get_broadcast_shape());
}

static bool cse_reshape(shared_ptr<GNode> a, shared_ptr<GNode> b)
{
    const auto reshape_a = std::static_pointer_cast<op::Reshape>(a->get_op_ptr());
    const auto reshape_b = std::static_pointer_cast<op::Reshape>(b->get_op_ptr());
    if (reshape_a == nullptr || reshape_b == nullptr)
    {
        return false;
    }
    return (reshape_a->get_input_order() == reshape_b->get_input_order()) &&
           (reshape_a->get_output_shape() == reshape_b->get_output_shape());
}

static unordered_map<std::string, function<bool(shared_ptr<GNode>, shared_ptr<GNode>)>>
    ops_to_cse_handlers = {{"Broadcast", cse_broadcast}, {"Reshape", cse_reshape}};

class NodeKey
{
public:
    NodeKey(const shared_ptr<GNode>& n)
        : m_node(n)
        , op_type(m_node->get_op_type())
    {
    }

    shared_ptr<GNode> get_node() const { return m_node; }
    bool operator==(const NodeKey& other) const
    {
        if (op_type == other.op_type)
        {
            const auto self_in = m_node->get_in_edges();
            const auto other_in = other.m_node->get_in_edges();
            bool same_inputs = (self_in.size() == other_in.size()) &&
                               std::equal(std::begin(self_in),
                                          std::end(self_in),
                                          std::begin(other_in),
                                          [](shared_ptr<nnfusion::graph::Edge> a,
                                             shared_ptr<nnfusion::graph::Edge> b) {
                                              return (a->get_src() == b->get_src()) &&
                                                     (a->get_src_output() == b->get_src_output());
                                          });
            if (same_inputs)
            {
                auto eh = ops_to_cse_handlers.find(op_type);
                if (eh != ops_to_cse_handlers.end())
                {
                    return eh->second(m_node, other.m_node);
                }
            }
        }

        return false;
    }

private:
    shared_ptr<GNode> m_node;
    std::string op_type;
};

namespace std
{
    template <>
    struct hash<NodeKey>
    {
        size_t operator()(const NodeKey& k) const
        {
            auto gnode = k.get_node();

            vector<size_t> arg_ids;
            hash<std::string> string_hash_compute{};
            hash<int> int_hash_compute{};

            auto op_type_hash = string_hash_compute(gnode->get_op_type());
            arg_ids.push_back(op_type_hash);

            for (auto edge : gnode->get_in_edges())
            {
                arg_ids.push_back(edge->get_src()->get_instance_id());
                arg_ids.push_back(edge->get_src_output());
            }

            auto hashc = hash_combine(arg_ids);
            return hashc;
        }
    };
}

bool CSEPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_cse = FLAGS_fcse;
    if (!enable_cse)
        return true;

    bool replaced = false;
    unordered_map<NodeKey, shared_ptr<GNode>> expressions{};

    for (auto n : graph->get_ordered_ops())
    {
        auto op = n->get_op_ptr();
        if (op->is_output() || op->is_parameter())
        {
            continue;
        }

        NodeKey n_key(n);
        if (expressions.count(n_key))
        {
            graph->replace_node(n, expressions.at(n_key), false);
            replaced = true;
        }
        else
        {
            expressions.insert(make_pair(n_key, n));
        }
    }
    return true;
}
