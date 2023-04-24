#include "if_single.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace std;
using namespace nnfusion::op;
using namespace nnfusion::graph;

std::shared_ptr<GNode> get_inner_fake_gnode(std::shared_ptr<graph::GNode> gnode, std::shared_ptr<op::Op> op) {
    GNodeIndexVector inner_inputs;
    auto in_edges = gnode->get_in_edges(); 
    for (auto it = in_edges.begin() + 1; it != in_edges.end(); ++it) {
        std::shared_ptr<graph::Edge> edge = *it;
        std::shared_ptr<graph::GNode> src = edge->get_src();
        int src_output = edge->get_src_output();
        inner_inputs.push_back(GNodeIndex(src, src_output));
    }
    std::shared_ptr<graph::GNode> inner_fake_node = std::make_shared<graph::GNode>(
        op, inner_inputs, gnode->get_output_size()
    );
    return inner_fake_node;
}

void IfSingle::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    std::shared_ptr<graph::GNode> inner_fake_node = get_inner_fake_gnode(gnode, m_inner_op);
    m_inner_fake_gnode = inner_fake_node;
    inner_fake_node->get_op_ptr()->revalidate_and_infer_types(inner_fake_node);
    for (size_t i = 0; i < gnode->get_output_size(); ++i) {
        gnode->set_output_type_and_shape(i, inner_fake_node->get_output_element_type(i), inner_fake_node->get_output_partial_shape(i));
    }
}

void IfSingle::infer_shared_memory(std::shared_ptr<graph::GNode> gnode) {
    NNFUSION_CHECK_NOT_NULLPTR(m_inner_fake_gnode);
    m_inner_op->infer_shared_memory(m_inner_fake_gnode);
    m_shared_memory = m_inner_op->get_shared_memory();
}