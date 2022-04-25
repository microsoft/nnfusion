#include "register_fusion_pass.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

#include "gflags/gflags.h"

#include <queue>

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

namespace
{
    struct TaggedNode
    {
        TaggedNode(shared_ptr<GNode> node, int id)
            : node_(node), id_(id)
        {
            dependency_count_ = node->get_input_size();
            visited_ = false;
            inlined_ = false;
            group_id_ = -1;
        }

        bool operator>(const TaggedNode &other) const { return id_ > other.id_; }

        std::shared_ptr<GNode> node_;
        int id_;
        int group_id_;
        size_t dependency_count_;
        bool visited_, inlined_;
    };
    struct FuseGroup
    {
        FuseGroup() {}
        std::unordered_set<shared_ptr<GNode>> nodes;
    };
    const std::unordered_set<std::string> inlined_ops = {"Broadcast", "Reshape"};
}

class RegisterFusionOptimizer {
public:
    RegisterFusionOptimizer(std::shared_ptr<Graph> g)
    : m_graph(g) {
        int id = 0;
        for (auto node : m_graph->get_ordered_ops()) {
            node_list_.push_back(make_shared<TaggedNode>(node, id++));
            node_map_[node] = node_list_.back();
        }
        update_inline_nodes();
        cur_group_ = 0;
    }

    bool Optimize() {
        for (auto& tnode : node_list_) {
            if (tnode->inlined_ || tnode->visited_)
                continue;
            // found a new reduce op
            fuse_from_node(tnode);
        }
        auto groups = extract_fusion_group();
        for (auto group: groups) {
            insert_fuse_group(group);
        }
        return true;
    }
private:
    vector<shared_ptr<FuseGroup>> extract_fusion_group() {
        unordered_map<int, shared_ptr<FuseGroup>> groups;
        vector<shared_ptr<FuseGroup>> result;
        for (auto& tnode : node_list_) {
            if (tnode->node_->get_op_ptr()->is_tensor_op() || tnode->group_id_ < 0) continue;
            if (!groups.count(tnode->group_id_)) {
                groups[tnode->group_id_] = make_shared<FuseGroup>();
            }
            groups[tnode->group_id_]->nodes.insert(tnode->node_);
        }
        for (auto& kv : groups) {
            if (kv.second->nodes.size() > 1) result.push_back(kv.second);
        }
        return result;
    }

    void insert_fuse_group(shared_ptr<FuseGroup> group) {
        auto fused_op = std::make_shared<nnfusion::op::Fused>("fused_kernel", "Matched_Pattern");
        auto fused_node = std::make_shared<FusedGNode>(fused_op);
        fused_node->build_fused_node(group->nodes, m_graph, true);
        m_graph->add_node(fused_node);
    }

    void fuse_from_node(shared_ptr<TaggedNode> &top_node) {
        unordered_set<shared_ptr<TaggedNode>> block_list;
        auto cmp = [](const shared_ptr<TaggedNode> &a, const shared_ptr<TaggedNode> &b) { return a->id_ > b->id_; };
        std::priority_queue<shared_ptr<TaggedNode>, vector<shared_ptr<TaggedNode>>, decltype(cmp)> queue(cmp);

        top_node->group_id_ = cur_group_++;
        queue.push(top_node);
        auto& output_shape = top_node->node_->get_output_shape(0);

        while (!queue.empty()) {
            auto tnode = queue.top();
            queue.pop();
            // std::cout << "process " <<  tnode->node_->get_op_type() << std::endl;
            if (block_list.count(tnode)) continue;
            auto& node = tnode->node_;
            NNFUSION_CHECK(node->get_output_size() == 1) << "Only support one output ops.";
            NNFUSION_CHECK(!tnode->visited_);

            // check fusible
            bool fusible = true;
            if (tnode != top_node) {
                fusible = (node->get_output_shape(0) == output_shape) && (tnode->inlined_);
            }

            // add to group
            if (fusible) {
                tnode->group_id_ = top_node->group_id_;
                for (auto& edge : node->get_out_edges()) {
                    queue.push(node_map_[edge->get_dst()]);
                }
                tnode->visited_ = true;
                if (tnode->inlined_) fuse_inline_dependent_nodes(tnode);
                update_inline_nodes();
            } else {
                update_block_list(block_list, tnode);
            }
        }
    }

    bool is_inlinable(std::shared_ptr<GNode> node) const {
        if (std::dynamic_pointer_cast<nnfusion::op::ElementwiseArithmetic>(node->get_op_ptr()))
            return true;
        if (inlined_ops.count(node->get_op_type()))
            return true;
        return false;
    }

    void fuse_inline_dependent_nodes(shared_ptr<TaggedNode> tnode) {
        for (auto edge : tnode->node_->get_in_edges()) {
            auto& in_node = node_map_[edge->get_src()];
            if (in_node->visited_) continue;
            NNFUSION_CHECK(in_node->inlined_);
            in_node->group_id_ = tnode->group_id_;
            in_node->visited_ = true;
            fuse_inline_dependent_nodes(in_node);
        }
    }

    void update_block_list(unordered_set<shared_ptr<TaggedNode>> &block_list, const shared_ptr<TaggedNode> &tnode) {
        for (auto edge : tnode->node_->get_out_edges()) {
            auto& out_node = node_map_[edge->get_dst()];
            if (block_list.count(out_node) == 0)
                update_block_list(block_list, out_node);
        }
    }

    void update_inline_nodes() {
        for (auto& tnode : node_list_) {
            if (tnode->inlined_ || tnode->visited_)
                continue;
            auto& node = tnode->node_;
            if (node->get_op_ptr()->is_tensor_op()) {
                tnode->inlined_ = true;
            } else if (is_inlinable(node)) {
                tnode->inlined_ = true;
                for (auto& edge : node->get_in_edges()) {
                    if (!(node_map_[edge->get_src()]->inlined_ || node_map_[edge->get_src()]->visited_)) {
                        tnode->inlined_ = false;
                        break;
                    }
                }
            }

        }
    }
    std::vector<shared_ptr<TaggedNode>> node_list_;
    std::unordered_map<shared_ptr<GNode>, shared_ptr<TaggedNode>> node_map_;
    shared_ptr<Graph> m_graph;
    int cur_group_;
};

bool RegisterFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto optimizer = RegisterFusionOptimizer(graph);
    return optimizer.Optimize();
}
