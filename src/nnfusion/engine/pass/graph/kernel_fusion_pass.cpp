// Microsoft (c) 2019, NNFusion Team

#include "kernel_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

#include "gflags/gflags.h"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_int32(fkernel_fusion_level, 2, "");
DECLARE_string(fdefault_device);

const static int DEFAULT_GROUP_ID = -1;

struct FuseGroup;
struct TaggedNode
{
    TaggedNode()
        : node(nullptr)
        , group_id(DEFAULT_GROUP_ID)
        , elem_group(nullptr)
        , ready_inputs(0)
        , visited(false)
    {
    }

    std::shared_ptr<GNode> node;
    int group_id;
    std::shared_ptr<FuseGroup> elem_group;
    size_t ready_inputs;
    bool visited;
};

struct FuseGroup
{
    FuseGroup(int g_id = DEFAULT_GROUP_ID)
        : id(g_id)
    {
    }
    int id;

    // <nodeid, <src_output_idx, in_slot_id>>
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> inputs;
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> consts;
    // <nodeid, <dst_input_idx, out_slot_id>>
    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<int, int>> outputs;

    size_t num_inputs;
    size_t num_consts;
    size_t num_outputs;

    size_t output_size = 0; // used in element_group

    std::vector<size_t> nodes;
};

class KernelFuseOptimizer
{
public:
    KernelFuseOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
        ELEM_GROUP_NODEID = m_nodes.size();
        RegisterFusionOps();
    }

    bool Optimize()
    {
        int fusion_level = FLAGS_fkernel_fusion_level;
        if (fusion_level > 0)
        {
            std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> fuse_groups =
                ExtractFusionGroups();
            if (fuse_groups != nullptr && fuse_groups->size() > 0)
            {
                if (fusion_level > 1)
                {
                    FuseReshapeAndBroadcast(fuse_groups);
                }
                // TagFusionGroupsOnGraph(fuse_groups);
                FuseElementGroupOnGraph(fuse_groups);
                return true;
            }
        }

        return true;
    }

private:
    void RegisterFusionOps()
    {
        static const std::vector<std::string> fuseop_list = {};
        //{"MatMul", "Split", "Concat", "ConcatV2", "Reshape"};

        static const std::vector<std::string> blacklist = {"Softmax"};

        fusion_ops.insert(fuseop_list.begin(), fuseop_list.end());
        op_blacklist.insert(blacklist.begin(), blacklist.end());

        // RegisterFusionOpFilters();
        host_inputs = {
            {"Split", 0}, {"Concat", 0}, {"ConcatV2", -1}, {"Reshape", 1}, {"ExpandDims", 1}};
    }

    void AddNodeToReadyQueues(std::shared_ptr<GNode> node,
                              std::queue<size_t>& ready,
                              std::deque<size_t>& fuse_ready,
                              std::deque<size_t>& elem_ready)
    {
        auto op = node->get_op_ptr();

        if (op_blacklist.count(node->get_op_type()) > 0)
        {
            ready.push(node->get_id());
            return;
        }

        if (std::dynamic_pointer_cast<nnfusion::op::ElementwiseArithmetic>(op))
        {
            elem_ready.push_front(node->get_id());
        }
        else if (fusion_ops.find(node->get_op_type()) != fusion_ops.end())
        {
            fuse_ready.push_front(node->get_id());
        }
        else
        {
            ready.push(node->get_id());
        }
    }

    size_t group_size(std::shared_ptr<FuseGroup> group)
    {
        if (!group)
            return 0;
        size_t num_nodes = 0;
        for (auto id : group->nodes)
        {
            CHECK(id < m_nodes.size());
            auto tn = m_nodes[id];
            if (id >= ELEM_GROUP_NODEID && tn->elem_group)
            {
                num_nodes += tn->elem_group->nodes.size();
            }
            else
            {
                num_nodes++;
            }
        }
        return num_nodes;
    }

    std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> ExtractFusionGroups()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups =
            std::make_shared<std::vector<std::shared_ptr<FuseGroup>>>();
        std::queue<size_t> ready;
        std::deque<size_t> fuse_ready;
        std::deque<size_t> elem_ready;
        enum WorkState
        {
            PROCESS_READY_NODE,
            PROCESS_FUSIABLE_NODE,
            PROCESS_ELEM_NODE,
            WORK_DONE
        };

        WorkState state = PROCESS_READY_NODE;
        std::shared_ptr<FuseGroup> cur_group = nullptr;
        std::shared_ptr<FuseGroup> cur_elemgroup = nullptr;

        for (auto node : m_graph->get_nodes())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
            }
        }

        while (state != WORK_DONE)
        {
            size_t node_id = 0;
            std::shared_ptr<TaggedNode> tn = nullptr;

            switch (state)
            {
            // Process the nodes in ready queue
            case PROCESS_READY_NODE:
            {
                if (ready.empty())
                {
                    state = (fuse_ready.empty() && elem_ready.empty()) ? WORK_DONE
                                                                       : PROCESS_FUSIABLE_NODE;
                    break;
                }

                node_id = ready.front();
                ready.pop();
                tn = m_nodes[node_id];
                break;
            }
            // Process the nodes in fuse_ready queue
            case PROCESS_FUSIABLE_NODE:
            {
                if (fuse_ready.empty())
                {
                    if (elem_ready.empty())
                    {
                        // Close the cur_group
                        if (cur_group && cur_group->nodes.size() > 0)
                        {
                            if (group_size(cur_group) > 1)
                            {
                                groups->push_back(cur_group);
                            }

                            cur_group = nullptr;
                        }
                        state = PROCESS_READY_NODE;
                    }
                    else
                    {
                        state = PROCESS_ELEM_NODE;
                    }
                    break;
                }
                node_id = fuse_ready.front();
                fuse_ready.pop_front();
                tn = m_nodes[node_id];

                if (!cur_group)
                    cur_group = std::make_shared<FuseGroup>();
                cur_group->nodes.push_back(node_id);
                break;
            }

            // Process the nodes in elem_ready queue
            case PROCESS_ELEM_NODE:
            {
                auto AppendElementGroup = [&]() {
                    if (cur_elemgroup && cur_elemgroup->nodes.size() > 0)
                    {
                        // append cur_elemgroup to cur_group
                        if (!cur_group)
                        {
                            cur_group = std::make_shared<FuseGroup>();
                        }
                        auto new_tn = std::make_shared<TaggedNode>();
                        new_tn->elem_group = cur_elemgroup;
                        int new_id = m_nodes.size();
                        m_nodes.push_back(new_tn);
                        cur_group->nodes.push_back(new_id);
                        cur_elemgroup = nullptr;
                    }
                };

                if (elem_ready.empty())
                {
                    AppendElementGroup();
                    state = PROCESS_FUSIABLE_NODE;
                    break;
                }

                node_id = elem_ready.front();
                elem_ready.pop_front();
                tn = m_nodes[node_id];

                size_t tensor_size = nnfusion::shape_size(tn->node->get_output_shape(0));
                if (cur_elemgroup && cur_elemgroup->output_size != tensor_size)
                {
                    AppendElementGroup();
                }

                if (!cur_elemgroup)
                {
                    cur_elemgroup = std::make_shared<FuseGroup>();
                }
                cur_elemgroup->nodes.push_back(node_id);
                cur_elemgroup->output_size = tensor_size;
                break;
            }

            // Do nothing
            case WORK_DONE: { break;
            }
            } // switch

            if (tn)
            {
                tn->visited = true;
                for (auto edge : tn->node->get_out_edges())
                {
                    auto dst = m_nodes[edge->get_dst()->get_id()];
                    dst->ready_inputs++;

                    if (!(dst->visited) && (dst->ready_inputs >= dst->node->get_in_edges().size()))
                    {
                        AddNodeToReadyQueues(dst->node, ready, fuse_ready, elem_ready);
                    }
                }
            }
        } // while

        return groups;
    }

    void FuseReshapeAndBroadcast(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        int next_fusion_group_id = 0;
        int next_elem_group_id = 0;

        for (auto group : *groups)
        {
            for (auto id : group->nodes)
            {
                CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group)
                {
                    std::vector<size_t> fusable_input_nodes;
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        CHECK(elem_id < m_nodes.size());
                        auto elem_tn = m_nodes[elem_id];
                        CHECK_NOT_NULLPTR(elem_tn->node);
                        for (auto in_edge : elem_tn->node->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                                continue;
                            auto input_node = in_edge->get_src();
                            while (input_node)
                            {
                                // A conservative way to avoid loop on DAG: only fuse node with single output
                                if (input_node->get_out_edges().size() > 1)
                                {
                                    break;
                                }
                                auto op = input_node->get_op_ptr();
                                if (auto bc =
                                        std::dynamic_pointer_cast<nnfusion::op::Broadcast>(op))

                                {
                                    if (bc->is_inner_broadcast() || bc->is_outer_broadcast())
                                    {
                                        fusable_input_nodes.push_back(input_node->get_id());
                                        // cannot fuse more nodes before broadcast
                                        break;
                                    }
                                }
                                else if (auto rs =
                                             std::dynamic_pointer_cast<nnfusion::op::Reshape>(op))
                                {
                                    if (!(rs->get_is_transpose()) &&
                                        (nnfusion::shape_size(input_node->get_output_shape(0)) ==
                                         nnfusion::shape_size(elem_tn->node->get_output_shape(0))))
                                        fusable_input_nodes.push_back(input_node->get_id());
                                }
                                else
                                {
                                    break;
                                }

                                bool input_set = false;
                                for (auto edge : input_node->get_in_edges())
                                {
                                    if (!edge->is_control_edge())
                                    {
                                        CHECK(input_set == false)
                                            << "Reshape and Broadcast can only have 1 input!";
                                        input_node = edge->get_src();
                                        input_set = true;
                                    }
                                }
                            }
                        }
                    }
                    // insert reshape and broadcast nodes into this element group
                    tn->elem_group->nodes.insert(tn->elem_group->nodes.begin(),
                                                 fusable_input_nodes.begin(),
                                                 fusable_input_nodes.end());
                }
            }
        }
    }

    void TagFusionGroupsOnGraph(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        int next_fusion_group_id = 0;
        int next_elem_group_id = 0;

        for (auto group : *groups)
        {
            // LOG(INFO) << DebugStringFuseGroup(group);
            for (auto id : group->nodes)
            {
                CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group)
                {
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        CHECK(elem_id < m_nodes.size());
                        auto elem_tn = m_nodes[elem_id];
                        CHECK_NOT_NULLPTR(elem_tn->node);

                        (*(elem_tn->node))["elem_group_id"] = next_elem_group_id;
                        (*(elem_tn->node))["fusion_group_id"] = next_fusion_group_id;
                    }
                    next_elem_group_id++;
                }
                else
                {
                    (*(tn->node))["fusion_group_id"] = next_fusion_group_id;
                }
            }
            next_fusion_group_id++;
        }
    }

    void FuseElementGroupOnGraph(std::shared_ptr<std::vector<std::shared_ptr<FuseGroup>>> groups)
    {
        for (auto group : *groups)
        {
            for (auto id : group->nodes)
            {
                CHECK(id < m_nodes.size());
                auto tn = m_nodes[id];
                if (id >= ELEM_GROUP_NODEID && tn->elem_group)
                {
                    std::vector<std::shared_ptr<KernelEmitter>> block_kernels;
                    bool all_kernel_emitted = true;
                    DeviceType dev_type;

                    // find and check whether all kernels are emitted
                    for (auto elem_id : tn->elem_group->nodes)
                    {
                        CHECK(elem_id < m_nodes.size());
                        auto node = m_nodes[elem_id]->node;
                        CHECK_NOT_NULLPTR(node);

                        auto emitted_kernels =
                            (*node)["Kernel_Selection_Result"]
                                .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                        auto emitter_iter =
                            find_if(emitted_kernels.begin(),
                                    emitted_kernels.end(),
                                    [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                        return (i.first == DeviceType::CUDA_GPU ||
                                                i.first == DeviceType::ROCM_GPU);
                                    });

                        KernelEmitter::Pointer kernel = nullptr;

                        if (emitter_iter == emitted_kernels.end() ||
                            emitter_iter->second == nullptr ||
                            emitter_iter->second->get_or_emit_source() == nullptr)
                        {
                            LOG(WARNING) << "Kernel should be emitted before this pass:"
                                         << node->get_name();
                            all_kernel_emitted = false;
                            break;
                        }
                        else
                        {
                            kernel = emitter_iter->second;
                            dev_type = emitter_iter->first;
                            block_kernels.push_back(kernel);
                        }
                    }

                    if (all_kernel_emitted)
                    {
                        auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                            "ElementWiseFused", CUDA_GPU, DT_FLOAT);
                        CHECK_NOT_NULLPTR(kernel_reg);
                        auto ctx = std::make_shared<KernelContext>();
                        ctx->kernels = block_kernels;
                        auto kernel = kernel_reg->m_factory(ctx);
                        kernel->get_or_emit_source();

                        //auto fused_node = std::make_shared<GNode>();
                        auto fused_op = std::make_shared<nnfusion::op::NoOp>("fused_kernel");
                        GNodeVector empty_inputs;
                        auto fused_node = std::make_shared<GNode>(fused_op, empty_inputs);
                        ctx->gnode = fused_node;

                        (*fused_node)["Kernel_Selection_Result"] =
                            vector<pair<DeviceType, KernelEmitter::Pointer>>();
                        auto& res = (*fused_node)["Kernel_Selection_Result"]
                                        .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                        res.push_back(std::make_pair(dev_type, kernel));

                        // replace original nodes with the fused node on graph
                        m_graph->add_node(fused_node);
                        int next_input_id = 0;
                        int next_output_id = 0;
                        std::unordered_set<std::shared_ptr<GNode>> internal_nodes;

                        for (auto elem_id : tn->elem_group->nodes)
                        {
                            auto node = m_nodes[elem_id]->node;
                            internal_nodes.insert(node);
                        }

                        for (auto elem_id : tn->elem_group->nodes)
                        {
                            auto node = m_nodes[elem_id]->node;
                            for (const auto& in_edge : node->get_in_edges())
                            {
                                if (internal_nodes.find(in_edge->get_src()) == internal_nodes.end())
                                {
                                    auto input_id = in_edge->is_control_edge() ? Graph::kControlSlot
                                                                               : next_input_id++;
                                    if (input_id != Graph::kControlSlot)
                                    {
                                        fused_node->set_input(
                                            input_id,
                                            node->get_inputs().at(in_edge->get_dst_input()));
                                    }
                                    m_graph->add_edge(in_edge->get_src(),
                                                      in_edge->get_src_output(),
                                                      fused_node,
                                                      input_id);
                                }
                            }

                            for (const auto& out_edge : node->get_out_edges())
                            {
                                if (internal_nodes.find(out_edge->get_dst()) ==
                                    internal_nodes.end())
                                {
                                    auto output_id = out_edge->is_control_edge()
                                                         ? Graph::kControlSlot
                                                         : next_output_id++;
                                    if (output_id != Graph::kControlSlot)
                                    {
                                        fused_node->set_output(
                                            output_id,
                                            node->get_outputs().at(out_edge->get_src_output()));
                                    }
                                    m_graph->add_edge(fused_node,
                                                      output_id,
                                                      out_edge->get_dst(),
                                                      out_edge->get_dst_input());
                                }
                            }
                        }

                        // ROCm can only support maximum 70 args for single kernel
                        if (FLAGS_fdefault_device == "ROCm" &&
                            fused_node->get_in_edges().size() +
                                    fused_node->get_out_edges().size() >=
                                70)
                        {
                            m_graph->remove_node(fused_node);
                        }
                        else
                        {
                            for (auto node : internal_nodes)
                            {
                                m_graph->remove_node(node);
                            }
                        }
                    }
                }
            }
        }
    }

    std::string DebugStringFuseGroup(std::shared_ptr<FuseGroup> group)
    {
        std::ostringstream ret;
        ret << "========================Fusion Group =====================\n";

        auto PrintInfo = [this, &ret](const size_t id) {
            auto n = m_nodes[id];
            ret << id << " / " << n->node->get_id() << ":" << n->node->get_name() << "\t"
                << n->node->get_op_type() << "\n";
        };

        ret << "FUSING NODES: [\n";
        for (auto id : group->nodes)
        {
            if (id < ELEM_GROUP_NODEID)
            {
                PrintInfo(id);
            }
            else
            {
                ret << "((\n";
                for (auto eid : m_nodes[id]->elem_group->nodes)
                {
                    PrintInfo(eid);
                }
                ret << ")) \n\n";
            }
        }
        ret << "]\n";
        return ret.str();
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
    size_t ELEM_GROUP_NODEID;

    std::unordered_set<std::string> op_blacklist;
    std::unordered_set<std::string> fusion_ops;
    std::unordered_map<std::string, int> host_inputs;
};

bool KernelFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device;
    if (dev_name == "ROCm" || dev_name == "CUDA")
    {
        LOG(INFO) << "device: " << dev_name;
        KernelFuseOptimizer optimizer(graph);
        return optimizer.Optimize();
    }
    else
    {
        return false;
    }
}