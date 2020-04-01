// Microsoft (c) 2019, NNFusion Team

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "nnfusion/core/graph/gedge.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace nnfusion::graph;
using namespace nnfusion::op;

atomic<size_t> GNode::m_next_instance_id(0);

GNode::GNode()
    : m_id(-1)
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
        CHECK(input_gnodes.at(i)->get_output_size() == 1) << "Argument " << i
                                                          << input_gnodes.at(i)->get_op_type()
                                                          << " must produce exactly one value.";
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
    CHECK(n >= m_inputs.size()) << "shrinking " << m_inputs.size() << " to " << n;
    for (size_t i = m_inputs.size(); i < n; ++i)
    {
        m_inputs.emplace_back(std::make_shared<Input>(element::dynamic, PartialShape::dynamic()));
    }
}

void GNode::set_output_size(size_t n)
{
    CHECK(n >= m_outputs.size()) << "shrinking " << m_outputs.size() << " to " << n;
    for (size_t i = m_outputs.size(); i < n; ++i)
    {
        auto tensor =
            make_shared<descriptor::Tensor>(element::dynamic,
                                            PartialShape::dynamic(),
                                            m_op_ptr->get_unique_name() + "_" + to_string(i));
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

const std::string& GNode::get_unique_name() const
{
    return m_unique_name;
}

void GNode::set_name(const string& name)
{
    m_name = name;
}

GNode::~GNode()
{
}

size_t GNode::set_id(size_t id)
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

const std::set<std::shared_ptr<nnfusion::graph::Edge>>& GNode::get_in_edges() const
{
    return m_in_edges;
};

const std::shared_ptr<nnfusion::graph::Edge> GNode::get_in_edge(size_t i) const
{
    CHECK(i < m_inputs.size()) << "Input index " << i << " is out of range. GNode only has "
                               << m_inputs.size() << " inputs.";
    std::shared_ptr<nnfusion::graph::Edge> found_in_edge = nullptr;
    for (auto in_edge : m_in_edges)
    {
        if (in_edge->get_dst_input() == i)
        {
            CHECK(found_in_edge == nullptr)
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

const std::set<std::shared_ptr<nnfusion::graph::Edge>>& GNode::get_out_edges() const
{
    return m_out_edges;
};

std::vector<std::shared_ptr<nnfusion::graph::Edge>> GNode::get_output_users(size_t i)
{
    CHECK(i < m_outputs.size()) << "Output index " << i << " is out of range. GNode only has "
                                << m_outputs.size() << " outputs.";
    std::vector<std::shared_ptr<nnfusion::graph::Edge>> output_users;

    auto edges = this->get_out_edges();
    for (auto edge : edges)
    {
        if (edge->get_src_output() == i)
        {
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
    CHECK(i < m_outputs.size()) << "Output index " << i << " is out of range. GNode only has "
                                << m_outputs.size() << " outputs.";

    return m_outputs.at(i)->get_tensor();
}

std::shared_ptr<nnfusion::descriptor::Tensor> GNode::get_output_tensor_ptr(size_t i) const
{
    CHECK(i < m_outputs.size()) << "Output index " << i << " is out of range. GNode only has "
                                << m_outputs.size() << " outputs.";

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
    CHECK(get_output_size() == 1)
        << "get_shape() must be called on a node with exactly one output.";
    return m_outputs.at(0)->get_shape();
}

const nnfusion::element::Type& GNode::get_element_type() const
{
    CHECK(get_output_size() == 1)
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
    m_id = -1;
    m_name.clear();
    m_op_type.clear();
}
