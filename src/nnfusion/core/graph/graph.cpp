// Microsoft (c) 2019, NNFusion Team

#include <sstream>

#include "graph.hpp"
#include "graph_util.hpp"
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
    CHECK(m_name.empty()) << "Graph name may be set exactly once.";

    m_name = name;
}

void Graph::add_node(std::shared_ptr<GNode> node)
{
    const size_t id = m_nodes.size();
    node->set_id(id);
    m_nodes.push_back(node);
    ++m_node_size;
}

std::shared_ptr<GNode> Graph::add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                const GNodeVector& input_gnodes)
{
    auto gnode = std::make_shared<GNode>(op, input_gnodes);

    add_node(gnode);

    for (size_t i = 0; i < input_gnodes.size(); i++)
    {
        add_edge(input_gnodes[i], 0, gnode, i);
    }
    op->revalidate_and_infer_types(gnode->shared_from_this());
    return gnode;
}

std::shared_ptr<GNode> Graph::add_node_and_edge(const std::shared_ptr<nnfusion::op::Op> op,
                                                const GNodeIndexVector& input_gnodes)
{
    auto gnode = std::make_shared<GNode>(op, input_gnodes);

    add_node(gnode);

    for (size_t i = 0; i < input_gnodes.size(); i++)
    {
        add_edge(input_gnodes[i].gnode, input_gnodes[i].index, gnode, i);
    }
    op->revalidate_and_infer_types(gnode->shared_from_this());
    return gnode;
}

void Graph::add_gnode_and_edge(const std::shared_ptr<GNode> gnode,
                               const GNodeIndexVector& input_gnodes)
{
    CHECK(gnode == nullptr) << "GNode can't be nullptr.";

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
    //DCHECK(!node->IsSource());
    //DCHECK(!node->IsSink());

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
    GNodeVector update_nodes;
    for (auto node : nodes)
    {
        if (node->get_op_type() == "Constant")
        {
            update_nodes.push_back(node);
        }
    }
    for (auto node : nodes)
    {
        if (node->get_op_type() != "Constant")
        {
            update_nodes.push_back(node);
        }
    }
    return update_nodes;
}

GNodeVector Graph::get_const_nodes()
{
    GNodeVector const_nodes;
    for (auto node : get_nodes())
    {
        if (node->get_op_type() == "Constant")
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
        CHECK(x == kControlSlot);
        CHECK(y == kControlSlot);
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
    CHECK(edge == m_edges[edge->m_id]);
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
        if (node != nullptr && node->get_op_ptr()->is_parameter())
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
