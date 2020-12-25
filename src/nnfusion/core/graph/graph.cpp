// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <sstream>

#include "graph.hpp"
#include "graph_util.hpp"
#include "nnfusion/common/serialize/attr_value.pb.h"
#include "nnfusion/common/serialize/graph_def.pb.h"
#include "nnfusion/common/serialize/pbtypes.pb.h"
#include "nnfusion/common/serialize/tensor_shape.pb.h"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::graph;

std::atomic<size_t> Graph::m_next_instance_id(0);

Graph::Graph(const std::string& name)
    : m_instance_id(m_next_instance_id.fetch_add(1))
    , m_temporary_pool_size(0)
    , m_name(name)
    , m_unique_name("Graph_" + std::to_string(m_instance_id))
{
    // TODO: need add source to sink control edge??
}

Graph::~Graph()
{
    // TODO: release node
}

const std::string& Graph::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Graph::get_name() const
{
    return m_unique_name;
}

void Graph::set_name(const std::string& name)
{
    NNFUSION_CHECK(m_name.empty()) << "Graph name may be set exactly once.";

    m_name = name;
}

void Graph::add_node(std::shared_ptr<GNode> node)
{
    if (node->get_id() != freeGnodeId)
    {
        // already added to graph
        NNFUSION_CHECK(m_nodes[node->get_id()] == node);
        return;
    }
    const size_t id = m_nodes.size();
    node->set_id(id);
    m_nodes.push_back(node);
    ++m_node_size;
}

std::shared_ptr<GNode> Graph::add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                const GNodeVector& input_gnodes,
                                                const size_t output_size)
{
    auto gnode = std::make_shared<GNode>(op, input_gnodes, output_size);

    add_node(gnode);

    for (size_t i = 0; i < input_gnodes.size(); i++)
    {
        add_edge(input_gnodes[i], 0, gnode, i);
    }
    op->revalidate_and_infer_types(gnode->shared_from_this());
    op->infer_shared_memory(gnode->shared_from_this());
    return gnode;
}

std::shared_ptr<GNode> Graph::add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                const GNodeIndexVector& input_gnodes,
                                                const size_t output_size)
{
    auto gnode = std::make_shared<GNode>(op, input_gnodes, output_size);

    add_node(gnode);

    for (size_t i = 0; i < input_gnodes.size(); i++)
    {
        add_edge(input_gnodes[i].gnode, input_gnodes[i].index, gnode, i);
    }
    op->revalidate_and_infer_types(gnode->shared_from_this());
    op->infer_shared_memory(gnode->shared_from_this());
    return gnode;
}

void Graph::add_gnode_and_edge(const std::shared_ptr<GNode> gnode,
                               const GNodeIndexVector& input_gnodes)
{
    NNFUSION_CHECK(gnode == nullptr) << "GNode can't be nullptr.";

    add_node(gnode);

    for (size_t i = 0; i < input_gnodes.size(); i++)
    {
        add_edge(input_gnodes[i].gnode, input_gnodes[i].index, gnode, i);
    }
    gnode->get_op_ptr()->revalidate_and_infer_types(gnode->shared_from_this());
}

void Graph::remove_node(std::shared_ptr<GNode> node)
{
    //TF_DCHECK_OK(IsValidNode(node)) << node->DebugString();
    //NNFUSION_DCHECK(!node->IsSource());
    //NNFUSION_DCHECK(!node->IsSink());

    // Remove any edges involving this node.
    while (!node->get_in_edges().empty())
    {
        remove_edge(*node->get_in_edges().begin());
    }
    while (!node->get_out_edges().empty())
    {
        remove_edge(*node->get_out_edges().begin());
    }
    m_nodes[node->get_id()] = nullptr;
    node->Clear();
    m_free_nodes.push_back(node);
    --m_node_size;
}

void Graph::replace_node(std::shared_ptr<GNode> old_node,
                         std::shared_ptr<GNode> new_node,
                         bool copy_in_edge)
{
    add_node(new_node);

    if (copy_in_edge)
    {
        for (auto& edge : old_node->get_in_edges())
        {
            if (edge->is_control_edge())
            {
                add_control_edge(edge->get_src(), new_node);
            }
            else
            {
                add_edge(edge->get_src(), edge->get_src_output(), new_node, edge->get_dst_input());
            }
        }
    }
    new_node->get_op_ptr()->revalidate_and_infer_types(new_node->shared_from_this());

    for (auto& edge : old_node->get_out_edges())
    {
        if (edge->is_control_edge())
        {
            add_control_edge(new_node, edge->get_dst());
        }
        else
        {
            add_edge(new_node, 0, edge->get_dst(), edge->get_dst_input());
        }
    }

    remove_node(old_node);
}

GNodeVector Graph::get_nodes()
{
    GNodeVector valid_nodes;
    for (auto node : m_nodes)
    {
        if (node != nullptr)
        {
            valid_nodes.push_back(node);
        }
    }
    return valid_nodes;
}

GNodeVector Graph::get_ordered_ops(bool include_control_deps)
{
    // todo: stored ops instead of calculate each time
    GNodeVector nodes;
    ReverseDFS(this,
               get_outputs(),
               nullptr,
               [&](std::shared_ptr<GNode> node) { nodes.push_back(node); },
               NodeComparatorName());

    return nodes;
}

GNodeVector Graph::get_bfs_ordered_ops()
{
    if (!m_bfs_ordered_ops_is_valid)
    {
        m_bfs_ordered_ops.clear();
        GNodeVector start;
        std::unordered_set<size_t> codegen_ops;
        for (auto node : get_ordered_ops())
        {
            codegen_ops.insert(node->get_id());
            if (node->get_input_size() == 0)
                start.push_back(node);
        }

        BFS(this,
            start,
            [&](std::shared_ptr<GNode> node) {
                if (codegen_ops.find(node->get_id()) != codegen_ops.end())
                    m_bfs_ordered_ops.push_back(node);
            },
            nullptr,
            NodeComparatorName());
        m_bfs_ordered_ops_is_valid = true;
    }
    return m_bfs_ordered_ops;
}

GNodeVector Graph::get_const_nodes()
{
    GNodeVector const_nodes;
    for (auto node : get_nodes())
    {
        if (node->is_constant())
        {
            const_nodes.push_back(node);
        }
    }
    return const_nodes;
}

const std::shared_ptr<nnfusion::graph::Edge>
    Graph::add_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y)
{
    //TF_DCHECK_OK(IsValidNode(source)) << source->DebugString();
    //TF_DCHECK_OK(IsValidNode(dest)) << dest->DebugString();

    //// source/sink must only be linked via control slots, and
    //// control slots must only be linked to control slots.
    //if (source == source_node() || dest == sink_node() || x == kControlSlot ||
    //y == kControlSlot) {
    //DCHECK_EQ(x, kControlSlot) << source->DebugString();
    //DCHECK_EQ(y, kControlSlot) << dest->DebugString();
    //}

    // control slots must only be linked to control slots
    if (x == kControlSlot || y == kControlSlot)
    {
        NNFUSION_CHECK(x == kControlSlot);
        NNFUSION_CHECK(y == kControlSlot);
    }
    else
    {
        auto source_type = source->get_output_element_type(x);
        auto dest_type = dest->get_input_element_type(y);
        NNFUSION_CHECK(source_type == dest_type)
            << "Fail to add edge, the soutce element type (" << source_type
            << " ) does not match the dest element type (" << dest_type << ").";

        auto source_shape = source->get_output_partial_shape(x);
        auto dest_shape = dest->get_input_partial_shape(y);
        NNFUSION_CHECK(source_shape.compatible(dest_shape))
            << "Fail to add edge, the source shape (" << source_shape
            << ") does not match the dest shape (" << dest_shape << ").";
    }

    std::shared_ptr<Edge> edge = nullptr;

    if (m_free_edges.empty())
    {
        edge = std::make_shared<Edge>(); // placement new
    }
    else
    {
        edge = m_free_edges.back();
        m_free_edges.pop_back();
    }
    edge->m_id = m_edges.size();
    edge->m_src = source;
    edge->m_dst = dest;
    edge->m_src_output = x;
    edge->m_dst_input = y;
    source->add_out_edge(edge);
    dest->add_in_edge(edge);
    m_edges.push_back(edge);

    ++m_edge_size;
    return edge;
}

bool Graph::find_edge(std::shared_ptr<GNode> source, int x, std::shared_ptr<GNode> dest, int y)
{
    for (const auto edge : m_edges)
    {
        if (edge->get_src() == source && edge->get_dst() == dest && edge->get_src_output() == x &&
            edge->get_dst_input() == y)
        {
            return true;
        }
    }
    return false;
}

const std::shared_ptr<nnfusion::graph::Edge> Graph::add_control_edge(std::shared_ptr<GNode> source,
                                                                     std::shared_ptr<GNode> dest,
                                                                     bool allow_duplicates)
{
    if (!allow_duplicates)
    {
        for (const auto edge : dest->get_in_edges())
        {
            if (edge->is_control_edge() && edge->get_src() == source)
            {
                // the requested edge already exist
                return nullptr;
            }
        }
    }

    return add_edge(source, kControlSlot, dest, kControlSlot);
}

void Graph::remove_edge(std::shared_ptr<Edge> edge)
{
    //TF_DCHECK_OK(IsValidNode(e->src_)) << e->src_->DebugString();
    //TF_DCHECK_OK(IsValidNode(e->dst_)) << e->dst_->DebugString();
    if (edge->is_control_edge())
    {
        // todo remove ^src from dst's node_def's input
    }
    edge->get_src()->remove_out_edge(edge);
    edge->get_dst()->remove_in_edge(edge);
    NNFUSION_CHECK(edge == m_edges[edge->m_id]);
    //CHECK_GT(m_num_edges, 0);

    m_edges[edge->m_id] = nullptr;

    edge->m_src = nullptr;
    edge->m_dst = nullptr;
    edge->m_id = -1;
    edge->m_src_output = kControlSlot - 1;
    edge->m_dst_input = kControlSlot - 1;
    m_free_edges.push_back(edge);
    --m_edge_size;
}

void Graph::set_default_outputs()
{
    m_output_nodes.clear();
    for (auto node : m_nodes)
    {
        if (node != nullptr && node->get_out_edges().size() == 0)
        {
            m_output_nodes.push_back(node);
        }
    }
}
void Graph::set_outputs(const GNodeVector& outputs)
{
    m_output_nodes = outputs;
}

GNodeVector Graph::get_outputs()
{
    return m_output_nodes;
}

const size_t Graph::get_output_size()
{
    return m_output_nodes.size();
}

const std::shared_ptr<GNode> Graph::get_output_op(size_t i)
{
    return m_output_nodes.at(i);
}

void Graph::set_default_parameters()
{
    m_parameters.clear();
    for (auto node : m_nodes)
    {
        if (node != nullptr && node->is_parameter())
        {
            m_parameters.push_back(node);
        }
    }
}

GNodeVector Graph::get_parameters()
{
    if (m_parameters.empty())
    {
        set_default_parameters();
    }
    return m_parameters;
}

size_t Graph::get_temporary_pool_size()
{
    return m_temporary_pool_size;
}

void Graph::set_temporary_pool_size(size_t size)
{
    m_temporary_pool_size = size;
}

bool Graph::serialize_to_file(const std::string& file_path)
{
    nnfusion::serialize::GraphDef graphdef;
    auto nnfusion_nodes = get_ordered_ops(true);
    for (auto& nnfusion_node : nnfusion_nodes)
    {
        NNFUSION_CHECK(
            !nnfusion_node->hasAttributes() ||
            (nnfusion_node->attributeNames().size() == 1 && nnfusion_node->hasAttribute("Alias")))
            << nnfusion_node->get_name() << " has " << nnfusion_node->attributeNames().size()
            << " tags including \"Alias\" which cannot be serialized now.";
        nnfusion::serialize::NodeDef* node = graphdef.add_node();
        // name
        node->set_name(nnfusion_node->get_name());
        // op
        node->set_op(nnfusion_node->get_op_type());
        // input
        for (auto nnfusion_edge : nnfusion_node->get_in_edges())
        {
            if (nnfusion_edge->get_src_output() == kControlSlot)
            {
                node->add_input("^" + nnfusion_edge->get_src()->get_name());
            }
            else
            {
                node->add_input(nnfusion_edge->get_src()->get_name() + ":" +
                                std::to_string(nnfusion_edge->get_src_output()));
            }
        }
        // TODO(gbxu): support all nnfusion ops
        if (nnfusion_node->get_op_type() == "AllReduce")
        {
            // tensor_name
            nnfusion::serialize::AttrValue tensor_name;
            tensor_name.set_s(nnfusion_node->get_name());
            (*node->mutable_attr())["tensor_name"] = tensor_name;
        }
        // data type
        if (nnfusion_node->get_output_size() == 1)
        {
            nnfusion::serialize::AttrValue data_type;
            nnfusion::serialize::PBType dt;
            nnfusion::element::Type::nnfusion_element_type_to_pbtype(
                nnfusion_node->get_element_type(), dt);
            data_type.set_type(dt);
            (*node->mutable_attr())["T"] = data_type;
        }
#if 0
        // Plan_gen can't parse this now. So just skip it for now.
        else
        {            
            nnfusion::serialize::AttrValue_ListValue* _data_types_list =
                new nnfusion::serialize::AttrValue_ListValue();
            for (auto nnfusion_output : nnfusion_node->get_outputs())
            {
                nnfusion::serialize::PBType dt;
                nnfusion::element::Type::nnfusion_element_type_to_pbtype(
                    nnfusion_output->get_element_type(), dt);
                _data_types_list->add_type(dt);
            }
            nnfusion::serialize::AttrValue _data_types;
            _data_types.set_allocated_list(_data_types_list);
            (*node->mutable_attr())["T"] = _data_types;
        }
#endif
        // _output_shapes
        nnfusion::serialize::AttrValue_ListValue* _output_shapes_list =
            new nnfusion::serialize::AttrValue_ListValue();
        for (auto nnfusion_output : nnfusion_node->get_outputs())
        {
            auto shape = _output_shapes_list->add_shape();
            for (auto nnfusion_dim : nnfusion_output->get_shape())
            {
                auto dim = shape->add_dim();
                dim->set_size(nnfusion_dim);
            }
        }
        nnfusion::serialize::AttrValue _output_shapes;
        _output_shapes.set_allocated_list(_output_shapes_list);
        (*node->mutable_attr())["_output_shapes"] = _output_shapes;
    }
    graphdef.set_version(1);
    std::fstream fs(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
    graphdef.SerializeToOstream(&fs);
    return true;
}

size_t Graph::get_memory_io()
{
    size_t total_io = 0;
    for (auto gnode : get_ordered_ops())
    {
        size_t node_io = 0;
        for (size_t i = 0; i < gnode->get_input_size(); ++i)
        {
            auto shape = gnode->get_input_shape(i);
            if (shape.size() > 0)
            {
                size_t ins = 1;
                for (auto d : shape)
                {
                    NNFUSION_CHECK(d != 0);
                    ins *= d;
                }
                node_io += ins;
            }
        }

        for (size_t i = 0; i < gnode->get_output_size(); ++i)
        {
            auto shape = gnode->get_output_shape(i);
            if (shape.size() > 0)
            {
                size_t outs = 1;
                for (auto d : shape)
                {
                    NNFUSION_CHECK(d != 0);
                    outs *= d;
                }
                node_io += outs;
            }
        }

        total_io += node_io;
    }

    return total_io;
}