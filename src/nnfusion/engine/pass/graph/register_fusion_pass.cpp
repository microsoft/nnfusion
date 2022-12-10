#include "register_fusion_pass.hpp"
#include "gflags/gflags.h"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"
#include "nnfusion/util/util.hpp"

#include <queue>

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_string(ftune_output_file, "", "the output json file path");
DEFINE_string(ftune_input_file, "", "the input json file path");
DEFINE_string(ffusion_skiplist, "", "List of op types that skips in fusion");
DECLARE_string(fdefault_device);

namespace
{
    struct TaggedNode
    {
        TaggedNode(shared_ptr<GNode> node, int id)
            : node_(node)
            , id_(id)
        {
            dependency_count_ = node->get_input_size();
            visited_ = false;
            inlined_ = false;
            group_id_ = -1;
        }

        bool operator>(const TaggedNode& other) const { return id_ > other.id_; }
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
    const std::unordered_set<std::string> inlined_ops = {
        "Broadcast", "Reshape", "Slice", "Convert", "CNHW2NCHW", "CNW2NCW", "HardSigmoid"};
    std::unordered_set<std::string> skip_ops = {};
    void parse_skip_ops()
    {
        stringstream ss(FLAGS_ffusion_skiplist);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            skip_ops.insert(substr);
        }
    }

    GNodeVector find_topo_sort_priority(std::shared_ptr<Graph> g)
    {
        GNodeVector nodes;
        unordered_map<shared_ptr<GNode>, int> topo_layer;
        ReverseDFS(g.get(),
                   g->get_outputs(),
                   [&](std::shared_ptr<GNode> node) { topo_layer[node] = 0; },
                   [&](std::shared_ptr<GNode> node) {
                       for (auto edge : node->get_in_edges())
                           topo_layer[node] =
                               max(topo_layer[node], topo_layer[edge->get_src()] + 1);
                   },
                   nullptr);
        ReverseDFS(g.get(),
                   g->get_outputs(),
                   nullptr,
                   [&](std::shared_ptr<GNode> node) { nodes.push_back(node); },
                   [&](std::shared_ptr<GNode> a, std::shared_ptr<GNode> b) {
                       return topo_layer[a] > topo_layer[b];
                   });
        return nodes;
    }
}

class RegisterFusionOptimizer
{
public:
    RegisterFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        int id = 0;
        for (auto node : find_topo_sort_priority(m_graph))
        {
            node_list_.push_back(make_shared<TaggedNode>(node, id++));
            node_map_[node] = node_list_.back();
        }
        update_inline_nodes();
        cur_group_ = 0;
    }

    bool Optimize()
    {
        for (auto& tnode : node_list_)
        {
            if (is_inlinable(tnode->node_) || tnode->visited_)
                continue;
            // found a new reduce op
            fuse_from_node(tnode);
        }
        for (auto& tnode : node_list_)
        {
            tnode->visited_ = false;
            tnode->inlined_ = false;
        }
        update_inline_nodes();
        for (auto& tnode : node_list_)
        {
            if (tnode->node_->get_op_ptr()->is_tensor_op() || tnode->visited_)
                continue;
            else if (tnode->group_id_ >= 0)
            { // already processed
                tnode->visited_ = true;
                update_inline_nodes();
            }
            else if (tnode->inlined_ && tnode->node_->get_out_edges().size() == 1 &&
                     node_map_[tnode->node_->get_out_edges()[0]->get_dst()]->group_id_ == -1)
            {
                // Inline these node to following nodes
                continue;
            }
            else
            {
                // fuse remaining elem op
                fuse_from_node(tnode, true);
            }
        }
        auto groups = extract_fusion_group();
        for (auto group : groups)
        {
            insert_fuse_group(group);
        }
        auto nodes = nlohmann::json().array();
        for (auto& node : find_topo_sort_priority(m_graph))
        {
            if (node->get_op_ptr()->is_tensor_op())
                continue;
            auto str = nnfusion::op::get_translation_v2(node);
            if (skip_ops.count(node->get_op_type()))
            {
                if (str.find("## @:") != string::npos)
                    str += "|skip";
                else
                    str += "## @: skip";
            }
            auto edge = nlohmann::json().array();
            for (auto& e : node->get_in_edges())
            {
                edge.push_back({e->get_src()->get_id(), e->get_src_output()});
            }
            string op_type = node->get_op_type();
            if (op_type == "Matched_Pattern")
                op_type = node->get_name();
            nodes.push_back({node->get_id(), str, op_type, edge});
        }
        auto file = std::ofstream(FLAGS_ftune_output_file);
        file << nodes.dump(/*indent=*/2);
        file.close();
        return true;
    }

private:
    vector<shared_ptr<FuseGroup>> extract_fusion_group()
    {
        unordered_map<int, shared_ptr<FuseGroup>> groups;
        vector<shared_ptr<FuseGroup>> result;
        for (auto& tnode : node_list_)
        {
            if (tnode->node_->get_op_ptr()->is_tensor_op() || tnode->group_id_ < 0)
                continue;
            if (!groups.count(tnode->group_id_))
            {
                groups[tnode->group_id_] = make_shared<FuseGroup>();
            }
            groups[tnode->group_id_]->nodes.insert(tnode->node_);
        }
        for (auto& kv : groups)
        {
            if (kv.second->nodes.size() > 1)
                result.push_back(kv.second);
        }
        return result;
    }

    void insert_fuse_group(shared_ptr<FuseGroup> group)
    {
        // get a meaningful name
        string name = "";
        map<int, string> values;
        for (auto node : group->nodes)
            values[node->get_id()] = node->get_op_type();
        for (auto pair : values)
        {
            if (name.size() > 0)
                name += "_";
            name += pair.second;
        }
        auto fused_op = std::make_shared<nnfusion::op::Fused>("fused_kernel", "Matched_Pattern");
        auto fused_node = std::make_shared<FusedGNode>(fused_op);
        fused_node->build_fused_node(group->nodes, m_graph, true);
        m_graph->add_node(fused_node);
        fused_node->set_name(name);
    }

    void fuse_from_node(shared_ptr<TaggedNode>& top_node, bool second = false)
    {
        unordered_set<shared_ptr<TaggedNode>> block_list;
        auto cmp = [](const shared_ptr<TaggedNode>& a, const shared_ptr<TaggedNode>& b) {
            return a->id_ > b->id_;
        };
        std::priority_queue<shared_ptr<TaggedNode>, vector<shared_ptr<TaggedNode>>, decltype(cmp)>
            queue(cmp);

        top_node->group_id_ = cur_group_++;
        queue.push(top_node);
        auto& output_shape = top_node->node_->get_output_shape(0);

        while (!queue.empty())
        {
            auto tnode = queue.top();
            queue.pop();
            // std::cout << "process " <<  tnode->node_->get_op_type() << std::endl;
            if (block_list.count(tnode))
                continue;
            auto& node = tnode->node_;
            NNFUSION_CHECK(node->get_output_size() == 1) << "Only support one output ops.";

            // check fusible
            bool fusible = true;
            if (tnode != top_node)
            {
                fusible &= tnode->group_id_ == -1;
                fusible &= !tnode->visited_;
                fusible &= tnode->inlined_;
                fusible &= node->get_output_shape(0) == output_shape;
                fusible &= !(skip_ops.count(node->get_op_type()) ||
                             skip_ops.count(top_node->node_->get_op_type()));
                if (node->get_op_type() == "Reshape")
                {
                    fusible &= !std::dynamic_pointer_cast<op::Reshape>(node->get_op_ptr())
                                    ->get_is_layout_change();
                }
            }

            // add to group
            if (fusible)
            {
                tnode->group_id_ = top_node->group_id_;
                for (auto& edge : node->get_out_edges())
                {
                    queue.push(node_map_[edge->get_dst()]);
                }
                tnode->visited_ = true;
                if (is_inlinable(tnode->node_))
                    fuse_inline_dependent_nodes(tnode);
                update_inline_nodes();
            }
            else
            {
                update_block_list(block_list, tnode);
            }
        }
    }

    bool is_inlinable(std::shared_ptr<GNode> node) const
    {
        if (std::dynamic_pointer_cast<nnfusion::op::ElementwiseArithmetic>(node->get_op_ptr()))
        {
            if (node->get_op_type() == "Softmax")
                return false;
            return true;
        }
        if (inlined_ops.count(node->get_op_type()))
            return true;
        return false;
    }

    void fuse_inline_dependent_nodes(shared_ptr<TaggedNode> tnode)
    {
        for (auto edge : tnode->node_->get_in_edges())
        {
            auto& in_node = node_map_[edge->get_src()];
            if (in_node->visited_)
                continue;
            if (!in_node->inlined_)
                continue;
            NNFUSION_CHECK(in_node->inlined_);
            in_node->group_id_ = tnode->group_id_;
            in_node->visited_ = true;
            fuse_inline_dependent_nodes(in_node);
        }
    }

    void update_block_list(unordered_set<shared_ptr<TaggedNode>>& block_list,
                           const shared_ptr<TaggedNode>& tnode)
    {
        block_list.insert(tnode);
        for (auto edge : tnode->node_->get_out_edges())
        {
            auto& out_node = node_map_[edge->get_dst()];
            if (block_list.count(out_node) == 0)
                update_block_list(block_list, out_node);
        }
    }

    void update_inline_nodes()
    {
        for (auto& tnode : node_list_)
        {
            if (tnode->inlined_ || tnode->visited_)
                continue;
            auto& node = tnode->node_;
            if (node->get_op_ptr()->is_tensor_op())
            {
                tnode->inlined_ = true;
            }
            else if (is_inlinable(node))
            {
                tnode->inlined_ = true;
                for (auto& edge : node->get_in_edges())
                {
                    if (!((node_map_[edge->get_src()]->inlined_ &&
                           edge->get_src()->get_out_edges().size() == 1) ||
                          node_map_[edge->get_src()]->visited_))
                    {
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

class ApplyFusionResult
{
public:
    ApplyFusionResult(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
    }
    bool apply(const string& fname)
    {
        auto fin = std::ifstream(fname, ios::in);
        json fusion_groups = json::parse(fin);
        NNFUSION_CHECK(fusion_groups.is_array());

        unordered_map<int, shared_ptr<GNode>> id2gnode;
        for (auto gnode : m_graph->get_ordered_ops())
            id2gnode[gnode->get_id()] = gnode;

        for (auto group : fusion_groups)
        {
            NNFUSION_CHECK(group.contains("nodes") && group["nodes"].is_array());
            if (!group.contains("code"))
                continue;

            std::vector<int> node_list;
            std::unordered_set<std::shared_ptr<GNode>> node_set;
            group["nodes"].get_to(node_list);
            for (auto node_id : node_list)
                node_set.insert(id2gnode[node_id]);

            // generates a meaningful name (for comments only)
            int group_id;
            group["group_id"].get_to(group_id);
            string name = "Group" + to_string(group_id);
            for (int node_id : node_list)
            {
                NNFUSION_CHECK(id2gnode.count(node_id));
                string op_type = id2gnode[node_id]->get_op_type();
                if (op_type == "Matched_Pattern")
                    op_type = id2gnode[node_id]->get_name();
                name += "  " + op_type;
            }
            auto fused_op = std::make_shared<nnfusion::op::Fused>(name, "GroupFusion");
            // handle inputs and outputs
            std::vector<pair<int, int>> input_desc, output_desc;
            group["input_desc"].get_to(input_desc);
            group["output_desc"].get_to(output_desc);
            auto fused_node = std::make_shared<GNode>();
            fused_node->construct_from_op_ptr(fused_op);
            for (int i = 0; i < input_desc.size(); i++)
            {
                auto node = id2gnode[input_desc[i].first];
                int in_id = input_desc[i].second;
                fused_node->set_input(i, node->get_inputs().at(in_id));
                auto edge = node->get_in_edges()[in_id];
                m_graph->add_edge(edge->get_src(), edge->get_src_output(), fused_node, i);
            }

            for (int i = 0; i < output_desc.size(); i++)
            {
                auto node = id2gnode[output_desc[i].first];
                int out_id = output_desc[i].second;
                fused_node->set_output(i, node->get_outputs().at(out_id));
                auto out_edges = node->get_output_users(out_id);
                for (auto out_edge : out_edges)
                {
                    auto out_node = out_edge->get_dst();
                    if (node_set.count(out_node))
                        continue;
                    m_graph->add_edge(fused_node, out_id, out_node, out_edge->get_dst_input());
                }
            }
            // cleanup
            for (auto& node : node_set)
                m_graph->remove_node(node);
            m_graph->add_node(fused_node);
            fused_node->set_name(name);
            shared_ptr<KernelContext> ctx(new KernelContext(fused_node));
            (*fused_node)["Kernel_Selection_Result"] =
                std::make_pair<NNFusion_DeviceType, KernelEmitter::Pointer>(
                    nnfusion::get_device_type(FLAGS_fdefault_device.c_str()),
                    make_shared<cuda::FusionCudaEmitter>(ctx, group));
        }
        return true;
    }

private:
    shared_ptr<Graph> m_graph;
};

bool RegisterFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_ftune_output_file == "")
        return true;
    NNFUSION_LOG(INFO) << "RegisterFusionPass Start";
    parse_skip_ops();
    auto optimizer = RegisterFusionOptimizer(graph);
    if (!optimizer.Optimize())
        return false;
    auto applier = ApplyFusionResult(graph);
    if (FLAGS_ftune_input_file == "")
        exit(0);
    applier.apply(FLAGS_ftune_input_file);
    NNFUSION_LOG(INFO) << "RegisterFusionPass Done";
    return true;
}
