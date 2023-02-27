// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

#include "nnfusion/core/operators/op_define/fused.hpp"

using namespace std;
using namespace nnfusion::graph;
using namespace nnfusion::op;

atomic<size_t> GNode::m_next_instance_id(0);

void GNode::reset_next_instance_id()
{
    m_next_instance_id = 0;
}

GNode::GNode()
    : m_id(Graph::freeGnodeId)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name("graph_node_" + to_string(m_instance_id))
{
}

GNode::GNode(const std::shared_ptr<Op> op_ptr, const GNodeVector& input_gnodes, size_t output_size)
    : GNode()
{
    initialize(op_ptr, input_gnodes, output_size);
}
GNode::GNode(const std::shared_ptr<Op> op_ptr,
             const GNodeIndexVector& input_gnode_indexs,
             size_t output_size)
    : GNode()
{
    initialize(op_ptr, input_gnode_indexs, output_size);
}

void GNode::construct_from_op_ptr(const std::shared_ptr<Op>& op_ptr)
{
    m_op_ptr = op_ptr;
    m_op_type = op_ptr->get_op_type();
    m_name = op_ptr->get_name();
}

void GNode::initialize(const std::shared_ptr<Op> op_ptr,
                       const GNodeVector& input_gnodes,
                       size_t output_size)
{
    construct_from_op_ptr(op_ptr);

    m_in_edges.clear();
    m_out_edges.clear();
    m_inputs.clear();
    m_outputs.clear();

    for (size_t i = 0; i < input_gnodes.size(); ++i)
    {
        /*
        NNFUSION_CHECK(input_gnodes.at(i)->get_output_size() == 1)
            << "Argument " << i << input_gnodes.at(i)->get_op_type()
            << " must produce exactly one value.";
            */
        m_inputs.emplace_back(
            std::make_shared<Input>(input_gnodes.at(i)->get_outputs().at(0)->get_element_type(),
                                    input_gnodes.at(i)->get_outputs().at(0)->get_partial_shape()));
    }

    set_output_size(output_size);
}

void GNode::initialize(const std::shared_ptr<Op> op_ptr,
                       const GNodeIndexVector& input_gnode_indexs,
                       size_t output_size)
{
    construct_from_op_ptr(op_ptr);

    m_in_edges.clear();
    m_out_edges.clear();
    m_inputs.clear();
    m_outputs.clear();

    for (size_t i = 0; i < input_gnode_indexs.size(); ++i)
    {
        m_inputs.emplace_back(std::make_shared<Input>(input_gnode_indexs.at(i)
                                                          .gnode->get_outputs()
                                                          .at(input_gnode_indexs.at(i).index)
                                                          ->get_element_type(),
                                                      input_gnode_indexs.at(i)
                                                          .gnode->get_outputs()
                                                          .at(input_gnode_indexs.at(i).index)
                                                          ->get_partial_shape()));
    }

    set_output_size(output_size);
}

void GNode::set_input_size(size_t n)
{
    NNFUSION_CHECK(n >= m_inputs.size()) << "shrinking " << m_inputs.size() << " to " << n;
    for (size_t i = m_inputs.size(); i < n; ++i)
    {
        m_inputs.emplace_back(std::make_shared<Input>(element::dynamic, PartialShape::dynamic()));
    }
}

void GNode::set_output_size(size_t n)
{
    NNFUSION_CHECK(n >= m_outputs.size()) << "shrinking " << m_outputs.size() << " to " << n;
    std::string loss_name;
    static std::unordered_map<std::string, int> used_name;

    if (get_name().find("loss") != string::npos)
    {
        loss_name = get_name();
        if (used_name.find(loss_name) != used_name.end())
        {
            loss_name = loss_name + "_" + to_string(used_name[loss_name]);
            used_name[get_name()] += 1;
        }
        else
        {
            used_name[loss_name] = 1;
        }
    }
    for (size_t i = m_outputs.size(); i < n; ++i)
    {
        auto tensor =
            make_shared<descriptor::Tensor>(element::dynamic,
                                            PartialShape::dynamic(),
                                            m_op_ptr->get_unique_name() + "_" + to_string(i));
        if (!(m_op_ptr->get_global_consistent_name().empty()))
        {
            tensor->set_global_consistent_name(m_op_ptr->get_global_consistent_name());
        }

        if (!loss_name.empty() && n == 1)
        {
            std::replace(loss_name.begin(), loss_name.end(), '/', '_');
            tensor->set_name(loss_name);
        }

        m_outputs.emplace_back(make_shared<Output>(tensor));
    }
}

const std::string& GNode::get_name() const
{
    if (m_name.empty())
    {
        return get_unique_name();
    }
    return m_name;
}

const std::string& GNode::get_member_name() const
{
    return m_member_name;
}

const std::string& GNode::get_unique_name() const
{
    return m_unique_name;
}

void GNode::set_name(const string& name)
{
    m_name = name;
}

void GNode::set_member_name(const string& name)
{
    m_member_name = name;
}

GNode::~GNode()
{
}

int64_t GNode::set_id(int64_t id)
{
    m_id = id;
    return m_id;
}

bool GNode::has_same_type(std::shared_ptr<const GNode> gnode) const
{
    if (get_output_size() != gnode->get_output_size())
    {
        return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i)
    {
        if (m_outputs[i]->get_element_type() != gnode->get_outputs().at(i)->get_element_type() ||
            m_outputs[i]->get_shape() != gnode->get_outputs().at(i)->get_shape())
        {
            return false;
        }
    }
    return true;
}

std::vector<std::shared_ptr<nnfusion::graph::Edge>> GNode::get_in_edges() const
{
    std::vector<std::shared_ptr<nnfusion::graph::Edge>> ret(m_in_edges.begin(), m_in_edges.end());
    std::sort(ret.begin(), ret.end(), EdgeComparatorDstIndex());
    return ret;
};

const std::shared_ptr<nnfusion::graph::Edge> GNode::get_in_edge(size_t i) const
{
    NNFUSION_CHECK(i < m_inputs.size()) << "Input index " << i
                                        << " is out of range. GNode only has " << m_inputs.size()
                                        << " inputs.";
    std::shared_ptr<nnfusion::graph::Edge> found_in_edge = nullptr;
    for (auto in_edge : m_in_edges)
    {
        if (in_edge->get_dst_input() == i)
        {
            NNFUSION_CHECK(found_in_edge == nullptr)
                << "There are more than one edges connect to Input index " << i;
            found_in_edge = in_edge;
        }
    }
    return found_in_edge;
}

void GNode::add_in_edge(std::shared_ptr<Edge> edge)
{
    m_in_edges.insert(edge);
}

void GNode::remove_in_edge(std::shared_ptr<nnfusion::graph::Edge> edge)
{
    m_in_edges.erase(edge);
}

std::vector<std::shared_ptr<nnfusion::graph::Edge>> GNode::get_out_edges() const
{
    std::vector<std::shared_ptr<nnfusion::graph::Edge>> ret(m_out_edges.begin(), m_out_edges.end());
    std::sort(ret.begin(), ret.end(), EdgeComparatorSrcIndex());
    return ret;
};

std::vector<std::shared_ptr<nnfusion::graph::Edge>>
    GNode::get_output_users(size_t i, bool include_control_edge)
{
    NNFUSION_CHECK(i < m_outputs.size()) << "Output index " << i
                                         << " is out of range. GNode only has " << m_outputs.size()
                                         << " outputs.";
    std::vector<std::shared_ptr<nnfusion::graph::Edge>> output_users;

    auto edges = this->get_out_edges();
    for (auto edge : edges)
    {
        if (edge->get_src_output() == i)
        {
            if (include_control_edge || !edge->is_control_edge())
                output_users.push_back(edge);
        }
    }
    return output_users;
}

void GNode::add_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge)
{
    m_out_edges.insert(edge);
}

void GNode::remove_out_edge(std::shared_ptr<nnfusion::graph::Edge> edge)
{
    m_out_edges.erase(edge);
}

nnfusion::descriptor::Tensor& GNode::get_input_tensor(size_t i) const
{
    auto in_edge = get_in_edge(i);
    return in_edge->get_src()->get_output_tensor(in_edge->get_src_output());
}

std::shared_ptr<nnfusion::descriptor::Tensor> GNode::get_input_tensor_ptr(size_t i) const
{
    auto in_edge = get_in_edge(i);
    return in_edge->get_src()->get_output_tensor_ptr(in_edge->get_src_output());
}

void GNode::set_input(size_t i, std::shared_ptr<Input> input)
{
    if (i >= m_inputs.size())
    {
        set_input_size(i + 1);
    }
    m_inputs[i] = input;
}

nnfusion::descriptor::Tensor& GNode::get_output_tensor(size_t i) const
{
    NNFUSION_CHECK(i < m_outputs.size()) << "Output index " << i
                                         << " is out of range. GNode only has " << m_outputs.size()
                                         << " outputs.";

    return m_outputs.at(i)->get_tensor();
}

std::shared_ptr<nnfusion::descriptor::Tensor> GNode::get_output_tensor_ptr(size_t i) const
{
    NNFUSION_CHECK(i < m_outputs.size()) << "Output index " << i
                                         << " is out of range. GNode only has " << m_outputs.size()
                                         << " outputs.";

    return m_outputs.at(i)->get_tensor_ptr();
}

void GNode::set_output(size_t i, std::shared_ptr<Output> output)
{
    if (i >= m_outputs.size())
    {
        set_output_size(i + 1);
    }
    m_outputs[i] = output;
}

void GNode::set_output_type_and_shape(size_t i,
                                      const nnfusion::element::Type& element_type,
                                      const nnfusion::PartialShape& pshape)
{
    if (i >= m_outputs.size())
    {
        set_output_size(i + 1);
    }
    m_outputs.at(i)->set_type_and_shape(element_type, pshape);
}

const nnfusion::Shape& GNode::get_shape() const
{
    NNFUSION_CHECK(get_output_size() == 1)
        << "get_shape() must be called on a node with exactly one output.";
    return m_outputs.at(0)->get_shape();
}

const nnfusion::element::Type& GNode::get_element_type() const
{
    NNFUSION_CHECK(get_output_size() == 1)
        << "get_element_type() must be called on a node with exactly one output.";
    return m_outputs.at(0)->get_element_type();
}

const nnfusion::element::Type& GNode::get_output_element_type(size_t i) const
{
    return m_outputs.at(i)->get_element_type();
}

const nnfusion::Shape& GNode::get_output_shape(size_t i) const
{
    return m_outputs.at(i)->get_shape();
}

const nnfusion::PartialShape& GNode::get_output_partial_shape(size_t i) const
{
    return m_outputs.at(i)->get_partial_shape();
}

const nnfusion::element::Type& GNode::get_input_element_type(size_t i) const
{
    return m_inputs.at(i)->get_element_type();
}

const nnfusion::Shape& GNode::get_input_shape(size_t i) const
{
    return m_inputs.at(i)->get_shape();
}

const nnfusion::PartialShape& GNode::get_input_partial_shape(size_t i) const
{
    return m_inputs.at(i)->get_partial_shape();
}

void GNode::Clear()
{
    m_in_edges.clear();
    m_out_edges.clear();
    m_inputs.clear();
    m_outputs.clear();
    m_op_ptr = nullptr;
    m_id = Graph::freeGnodeId;
    m_name.clear();
    m_op_type.clear();
}

const nnfusion::Shape& GNodeIndex::get_shape() const
{
    return gnode->get_output_shape(index);
}

const nnfusion::element::Type& GNodeIndex::get_element_type() const
{
    return gnode->get_output_element_type(index);
}

void FusedGNode::build_fused_node(std::unordered_set<std::shared_ptr<GNode>> nodes,
                                  std::shared_ptr<Graph> graph,
                                  bool clean_graph)
{
    reorder_nodes(nodes, graph);
    set_inputs_and_outputs(graph);
    derive_op_def();
    if (clean_graph)
        clean_nodes(graph);
}

void FusedGNode::reorder_nodes(std::unordered_set<std::shared_ptr<GNode>> nodes,
                               std::shared_ptr<Graph> graph)
{
    NNFUSION_CHECK(!m_order_nodes.size());
    for (auto& node : graph->get_ordered_ops())
    {
        if (nodes.find(node) == nodes.end())
            continue;
        m_order_nodes.push_back(node);
    }
}

void FusedGNode::set_inputs_and_outputs(std::shared_ptr<Graph> graph)
{
    NNFUSION_CHECK(m_order_nodes.size());
    std::unordered_set<std::shared_ptr<GNode>> cached_nodes;
    for (const auto& m_node : m_order_nodes)
    {
        cached_nodes.insert(m_node);
        auto ctx = std::make_shared<OpContext>(m_node);
        m_op_ctxs.push_back(ctx);
    }

    std::unordered_map<std::shared_ptr<GNode>, std::unordered_map<size_t, size_t>> input_id_map;
    // Register input tensors
    for (const auto& m_node : m_order_nodes)
    {
        // Add non-control-edges as inputs of fused node
        auto next_input_id = m_inputs.size();
        for (auto in_id = 0; in_id < m_node->get_input_size(); ++in_id)
        {
            auto& in_edge = m_node->get_in_edge(in_id);
            if (cached_nodes.find(in_edge->get_src()) == cached_nodes.end())
            {
                auto input_id = next_input_id++;
                set_input(input_id, m_node->get_inputs().at(in_edge->get_dst_input()));
                graph->add_edge(
                    in_edge->get_src(), in_edge->get_src_output(), shared_from_this(), input_id);
                input_id_map[m_node][in_edge->get_dst_input()] = input_id;
            }
        }
        // Add control-edges as inputs of fused node
        for (const auto& in_edge : m_node->get_in_edges())
        {
            if (!in_edge->is_control_edge())
                continue;
            graph->add_edge(in_edge->get_src(),
                            in_edge->get_src_output(),
                            shared_from_this(),
                            Graph::kControlSlot);
        }
    }

    // Register output tensors
    for (const auto& m_node : m_order_nodes)
    {
        for (int out_id = 0; out_id < m_node->get_output_size(); ++out_id)
        {
            bool has_output = false;
            auto out_edges = m_node->get_output_users(out_id);
            for (auto out_edge : out_edges)
            {
                auto out_node = out_edge->get_dst();
                if (cached_nodes.find(out_node) != cached_nodes.end())
                    continue;
                if (!has_output)
                {
                    has_output = true;
                    set_output(get_output_size(),
                               m_node->get_outputs().at(out_edge->get_src_output()));

                    // get inplace annotation
                    auto op = std::dynamic_pointer_cast<Op>(m_node->get_op_ptr());
                    auto op_annotations = op->get_op_annotations();
                    if (op_annotations)
                    {
                        auto oi_pairs = op_annotations->get_in_place_oi_pairs();
                        for (auto oi_pair : oi_pairs)
                        {
                            auto iter = input_id_map.find(m_node);
                            if (iter != input_id_map.end() && iter->second.count(oi_pair.input) > 0)
                            {
                                auto fused_op =
                                    std::dynamic_pointer_cast<Op>(shared_from_this()->get_op_ptr());
                                AddInplace(fused_op,
                                           get_output_size() - 1,
                                           iter->second[oi_pair.input],
                                           oi_pair.destructive,
                                           oi_pair.force_inplace);
                                //NNFUSION_LOG(INFO) << "========================: node=" << m_node->get_op_type() << ", oi: <" << oi_pair.output << ", " << oi_pair.input << ">";
                            }
                        }
                    }
                }
                graph->add_edge(shared_from_this(),
                                get_output_size() - 1,
                                out_edge->get_dst(),
                                out_edge->get_dst_input());
            }
        }
    }
}

void FusedGNode::clean_nodes(std::shared_ptr<Graph> graph)
{
    for (auto& node : m_order_nodes)
        graph->remove_node(node);
}

void FusedGNode::derive_op_def()
{
    static_pointer_cast<nnfusion::op::Fused>(m_op_ptr)->register_ir2(m_order_nodes,
                                                                     shared_from_this());
}
