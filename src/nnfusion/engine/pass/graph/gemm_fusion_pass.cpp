// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <climits>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gemm_fusion_pass.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DEFINE_bool(fgemm_fusion, false, "GEMM fusion.");

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

namespace
{
    struct TaggedNode
    {
        TaggedNode()
            : node(nullptr)
            , depth(INT_MAX)
            , groupid(-1)
        {
            depends.clear();
        }

        int depth;
        int groupid;
        std::unordered_set<int> depends;
        std::shared_ptr<GNode> node;
    };

    // A MatGroup is a group of "MatMul" nodes that may be merged.
    // By defalut, can_merge is true.
    struct MatGroup
    {
        MatGroup()
            : groupid(-1)
            , can_merge(true)
        {
            nodes.clear();
            sub_groups.clear();
        }

        int groupid;
        bool can_merge;
        std::vector<std::shared_ptr<GNode>> nodes;
        std::vector<std::vector<std::shared_ptr<GNode>>> sub_groups;
    };

    bool SortByName(std::shared_ptr<GNode> lhs, std::shared_ptr<GNode> rhs)
    {
        return lhs->get_name() < rhs->get_name();
    }
}

class GEMMFuseOptimizer
{
public:
    GEMMFuseOptimizer(std::shared_ptr<Graph> g, int rounds)
        : m_graph(g)
        , m_rounds(rounds)
        , m_next_node_id(0)
        , m_max_depth(100)
    {
    }

    bool Optimize()
    {
        auto gnodes = m_graph->get_ordered_ops();

        // Step 1: find all the MatMul operators and group them using their hash value.
        std::unordered_map<size_t, int> hash_to_gid;
        std::vector<std::shared_ptr<MatGroup>> merge_groups;

        hash_to_gid.clear();
        merge_groups.clear();

        bool changed = false;
        int groupid = 0;

        for (auto node : gnodes)
        {
            if (node->get_op_type() == "Dot")
            {
                size_t hash_value = NodeHash(node);
                if (hash_to_gid.count(hash_value) == 0)
                {
                    hash_to_gid[hash_value] = groupid;
                    std::shared_ptr<MatGroup> g(new MatGroup());
                    g->groupid = groupid;
                    merge_groups.push_back(g);
                    groupid++;
                }
                merge_groups[hash_to_gid[hash_value]]->nodes.push_back(node);
            }
        }

        // Step 2: extract the dependencies among all candidates of each group.
        std::unordered_map<int, TaggedNode> tagged_nodes;
        tagged_nodes.clear();

        for (auto group : merge_groups)
        {
            for (std::shared_ptr<GNode> node : group->nodes)
            {
                tagged_nodes[node->get_id()].node = node;
                tagged_nodes[node->get_id()].groupid = group->groupid;
            }
        }
        ExtractDependencies(tagged_nodes);

        // Step 3: for each candidate_group, partition the candidate nodes into
        // independent sub-groups, where each sub-group can be merged to one operator
        // (TODO) currently, we take a very naive partition method.
        for (auto group : merge_groups)
        {
            if (group->nodes.size() < 2)
            {
                group->can_merge = false;
                continue;
            }

            std::vector<bool> assigned(group->nodes.size(), false);
            std::vector<std::shared_ptr<GNode>> sub_group;

            for (size_t i = 0; i < group->nodes.size(); i++)
            {
                sub_group.clear();
                for (size_t j = i; j < group->nodes.size(); j++)
                {
                    if (!assigned[j])
                    {
                        int cand_id = group->nodes[j]->get_id();
                        bool can_be_fused = true;
                        for (auto nd : sub_group)
                        {
                            int src_id = nd->get_id();
                            if (std::abs(tagged_nodes[src_id].depth - tagged_nodes[cand_id].depth) >
                                    m_max_depth ||
                                tagged_nodes[src_id].depends.count(cand_id) != 0 ||
                                tagged_nodes[cand_id].depends.count(src_id) != 0)
                            {
                                can_be_fused = false;
                                break;
                            }
                        }

                        if (can_be_fused)
                        {
                            sub_group.push_back(group->nodes[j]);
                            assigned[j] = true;
                        }
                    }
                }

                if (sub_group.size() >= 2)
                {
                    group->sub_groups.push_back(sub_group);
                }
            }

            if (group->sub_groups.size() == 0)
            {
                group->can_merge = false;
            }
        }

        // Step4: for each sub-group that can be merged, we merge them into one
        // new "MatMul" operator with an additional "Concat" operator and a "Split" Operator.
        int merge_before = 0;
        int merge_after = 0;
        for (auto group : merge_groups)
        {
            for (auto subgroup : group->sub_groups)
            {
                if (MergeIntoOneInplace(subgroup))
                {
                    changed = true;
                    merge_before += subgroup.size();
                    merge_after++;
                }
            }
        }

        if (changed)
        {
            NNFUSION_LOG(INFO) << "[NNFusion] GEMM Fusion: round " << m_rounds << ", merged "
                               << merge_before << " Dot ops into " << merge_after;
        }

        return changed;
    }

private:
    size_t NodeHash(const std::shared_ptr<GNode> node)
    {
        // Concat the right operand in even rounds, and left in odd rounds.
        int input_idx = m_rounds % 2;

        std::shared_ptr<nnfusion::graph::GNode> src_node = nullptr;
        for (const auto in_edge : node->get_in_edges())
        {
            if (in_edge->get_dst_input() == input_idx)
            {
                src_node = in_edge->get_src();
                break;
            }
        }

        string str_to_hash = node->get_op_type() + src_node->get_name();
        std::size_t str_hash = std::hash<std::string>{}(str_to_hash);

        return str_hash;
    }

    void ExtractDependencies(std::unordered_map<int, TaggedNode>& inout)
    {
        // Use a BFS to propagate the node depth and the dependencies
        // we use node id to index a node.
        std::vector<TaggedNode> nodes(m_graph->get_max_node_id(), TaggedNode());

        // First initalize the known nodes form input, so that we
        // know their groupids.
        for (auto elem : inout)
        {
            nodes[elem.first] = elem.second;
        }

        std::queue<int> actives;
        std::vector<std::shared_ptr<GNode>> all_nodes = m_graph->get_nodes();

        // Push all the input nodes.
        for (auto node : all_nodes)
        {
            if (node != nullptr && node->get_input_size() == 0)
            {
                TaggedNode& tn = nodes[node->get_id()];
                if (!tn.node)
                    tn.node = node;
                actives.push(node->get_id());
            }
        }

        while (!actives.empty())
        {
            int id = actives.front();
            actives.pop();
            TaggedNode& n = nodes[id];

            for (auto outedge : n.node->get_out_edges())
            {
                std::shared_ptr<GNode> out = outedge->get_dst();
                TaggedNode& tn = nodes[out->get_id()];
                if (!tn.node)
                    tn.node = out;

                bool changed = false;
                if (n.depth + 1 < tn.depth)
                {
                    tn.depth = n.depth + 1;
                    changed = true;
                }
                int before = tn.depends.size();
                tn.depends.insert(n.depends.begin(), n.depends.end());

                // If id is in the candidate list, add the dependency of id.
                if (inout.count(id) > 0)
                {
                    tn.depends.insert(id);
                }
                if (tn.depends.size() > before)
                {
                    changed = true;
                }

                if (changed)
                {
                    actives.push(out->get_id());
                }
            }
        }

        // For each output node, only add its dependent nodes who are in the
        // same group.
        for (auto& id_node : inout)
        {
            for (auto depend : nodes[id_node.first].depends)
            {
                if (nodes[depend].groupid == id_node.second.groupid)
                {
                    id_node.second.depends.insert(depend);
                }
            }
        }
    }

    string GetNewNodeName(const string& prefix)
    {
        std::stringstream ss;
        ss << prefix << m_rounds << "_" << m_next_node_id++;
        return ss.str();
    }

    bool MergeIntoOneInplace(std::vector<std::shared_ptr<GNode>> nodes)
    {
        bool changed = false;
        int concat_size = nodes.size();
        if (concat_size < 2)
            return changed;

        std::sort(nodes.begin(), nodes.end(), SortByName);

        // Step 0: At this stage, the transpose attr of left operand and right operand
        // of all nodes in the same group shoud be same.
        // Otherwise, we need add another transpose operator, which may increase
        // the fusing overhead.
        GNodeIndexVector concat_inputs;

        std::vector<int> lengths;

        bool same_operand_transpose = false;
        int in_idx = (m_rounds + 1) % 2;

        for (auto gnode : nodes)
        {
            std::shared_ptr<GNode> src_gnode = nullptr;
            int src_output_idx = -1;
            for (auto edge : gnode->get_in_edges())
            {
                if (edge->get_dst_input() == in_idx)
                {
                    src_gnode = edge->get_src();
                    src_output_idx = edge->get_src_output();
                    break;
                }
            }
            if (src_gnode)
            {
                auto dot_op = std::dynamic_pointer_cast<nnfusion::op::Dot>(gnode->get_op_ptr());
                std::shared_ptr<GNode> reshape_node = nullptr;
                if (dot_op && in_idx == 1)
                {
                    same_operand_transpose = dot_op->get_transpose_A();
                    if (dot_op->get_transpose_B())
                    {
                        reshape_node =
                            numpy_transpose(src_gnode, nnfusion::AxisVector{1, 0}, src_output_idx);
                    }
                }
                else if (dot_op && in_idx == 0)
                {
                    same_operand_transpose = dot_op->get_transpose_B();
                    if (dot_op->get_transpose_A())
                    {
                        reshape_node =
                            numpy_transpose(src_gnode, nnfusion::AxisVector{1, 0}, src_output_idx);
                    }
                }

                nnfusion::Shape shape;
                if (reshape_node != nullptr)
                {
                    m_graph->add_gnode_and_edge(
                        reshape_node, GNodeIndexVector({GNodeIndex(src_gnode, src_output_idx)}));
                    concat_inputs.push_back(GNodeIndex(reshape_node, 0));
                    shape = reshape_node->get_shape();
                }
                else
                {
                    concat_inputs.push_back(GNodeIndex(src_gnode, src_output_idx));
                    shape = src_gnode->get_output_shape(src_output_idx);
                }

                if (in_idx == 1)
                {
                    lengths.push_back(shape[1]);
                }
                else
                {
                    lengths.push_back(shape[0]);
                }
            }
        }

        size_t concat_axis;

        NNFUSION_LOG(INFO) << "Num nodes before merge: " << m_graph->get_node_size();

        // Step 1: describe the concat_dim.
        if (in_idx == 1)
        {
            // Concat rhs.
            concat_axis = 1;
        }
        else
        {
            // Concat lhs.
            concat_axis = 0;
        }

        // Step 2: Add concat node
        // To do the inplace merge, we need to set an allocation indicator
        // for each input node of the MatMul node, so that we can make sure the input
        // tensors are placed in a continuous memory space.
        auto concat_op = std::make_shared<nnfusion::op::Concat>(concat_axis);
        concat_op->set_name(GetNewNodeName(std::string("gemm_fusion_concat_node_")));
        auto concat_gnode = m_graph->add_node_and_edge(concat_op, concat_inputs);

        // Step 3: add a new MatMul(Dot) node.
        std::shared_ptr<GNode> same_operand = nullptr;
        int same_operand_output_idx = -1;
        for (auto edge : nodes[0]->get_in_edges())
        {
            if (edge->get_dst_input() == m_rounds % 2)
            {
                same_operand = edge->get_src();
                same_operand_output_idx = edge->get_src_output();
                break;
            }
        }

        auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, same_operand_transpose, false);

        dot_op->set_name(GetNewNodeName(std::string("gemm_fusion_dot_node_")));
        GNodeIndexVector dot_inputs;

        if (m_rounds % 2 == 0)
        {
            dot_inputs.push_back(GNodeIndex(same_operand, same_operand_output_idx));
            dot_inputs.push_back(GNodeIndex(concat_gnode, 0));
        }
        else
        {
            dot_inputs.push_back(GNodeIndex(concat_gnode, 0));
            dot_inputs.push_back(GNodeIndex(same_operand, same_operand_output_idx));
        }
        auto dot_gnode = m_graph->add_node_and_edge(dot_op, dot_inputs);

        // Step 4: add slice node.
        nnfusion::Shape shape = dot_gnode->get_shape();
        int rank = shape.size();
        std::vector<size_t> lower(rank, 0);
        std::vector<size_t> upper(shape);
        int cursor = 0;
        int slice_axis = 0;
        if (in_idx == 1)
        {
            slice_axis = rank - 1;
        }

        for (int i = 0; i < lengths.size(); ++i)
        {
            lower[slice_axis] = cursor;
            cursor += lengths[i];
            upper[slice_axis] = cursor;

            auto slice_op = std::make_shared<nnfusion::op::Slice>(lower, upper);
            slice_op->set_name(GetNewNodeName(std::string("gemm_fusion_slice_node_")));
            auto slice_gnode = m_graph->add_node_and_edge(slice_op, {dot_gnode});

            // first check whether there is control edge for input edges
            for (auto edge : nodes[i]->get_in_edges())
            {
                if (edge->is_control_edge())
                {
                    m_graph->add_control_edge(edge->get_src(), concat_gnode);
                }
            }

            for (auto edge : nodes[i]->get_out_edges())
            {
                if (edge->is_control_edge())
                {
                    m_graph->add_control_edge(slice_gnode, edge->get_dst());
                }
                else
                {
                    m_graph->add_edge(slice_gnode, 0, edge->get_dst(), edge->get_dst_input());
                }
            }
        }

        // Now can safely remove each Matmul node.
        for (auto node : nodes)
        {
            m_graph->remove_node(node);
        }

        NNFUSION_LOG(INFO) << "Num nodes after merge: " << m_graph->get_node_size();
        return true;
    }

private:
    std::shared_ptr<Graph> m_graph;
    int m_rounds;
    int m_next_node_id;
    int m_max_depth;
};

bool GemmFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    bool enable_gemm_fusion = FLAGS_fgemm_fusion;
    if (!enable_gemm_fusion)
        return true;

    const int kMaxRounds = 10;
    bool changed = true;

    for (int rounds = 0; rounds < kMaxRounds; ++rounds)
    {
        changed = false;

        GEMMFuseOptimizer optimizer(graph, rounds);

        if (optimizer.Optimize())
        {
            changed = true;
        }

        if (!changed)
            break;
    }
    return true;
}
