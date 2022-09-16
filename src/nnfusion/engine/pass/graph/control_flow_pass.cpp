// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "control_flow_pass.hpp"

#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
#include "nnfusion/engine/device/cuda.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::engine;
using Loop = nnfusion::op::Loop;
using If = nnfusion::op::If;
using Recursion = nnfusion::op::Recursion;

// this func append Constants in the subgraph as additional inputs to the control flow node
static void extract_constant_nodes(std::shared_ptr<nnfusion::graph::Graph>& graph,
                                   std::shared_ptr<nnfusion::graph::GNode> gnode,
                                   const ir::Program& program)
{
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto node = ins->getGNode();
            if (node->get_op_type() == "Constant")
            {
                auto new_node = make_shared<nnfusion::graph::GNode>(node->get_op_ptr(),
                                                                    nnfusion::graph::GNodeVector());
                new_node->set_output(0,
                                     make_shared<nnfusion::graph::Output>(ins->get_outputs()[0]));
                auto idx = gnode->get_input_size();
                node->Set<int>("subgraph_input_map", idx);
                gnode->set_input(idx,
                                 make_shared<nnfusion::graph::Input>(node->get_element_type(),
                                                                     node->get_shape()));
                graph->add_node(new_node);
                graph->add_edge(new_node, 0, gnode, idx);
            }
        }
    }
}

bool ControlFlowPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    for (auto gnode : graph->get_nodes())
    {
        if (gnode->get_op_type() == "Loop")
        {
            auto op = static_pointer_cast<Loop>(gnode->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(op);
            auto loop_body = op->get_loop_body_graph();
            auto loop_body_tu = CudaEngine().convert_graph_to_program(loop_body, false);
            op->set_loop_body_tu(loop_body_tu);
            extract_constant_nodes(graph, gnode, loop_body_tu->program);
        }
        else if (gnode->get_op_type() == "If")
        {
            auto op = static_pointer_cast<If>(gnode->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(op);
            auto then_branch = op->get_then_branch_graph();
            auto else_branch = op->get_else_branch_graph();
            auto then_branch_tu = CudaEngine().convert_graph_to_program(then_branch, false);
            auto else_branch_tu = CudaEngine().convert_graph_to_program(else_branch, false);
            op->set_then_branch_tu(then_branch_tu);
            op->set_else_branch_tu(else_branch_tu);
            extract_constant_nodes(graph, gnode, then_branch_tu->program);
            extract_constant_nodes(graph, gnode, else_branch_tu->program);
        }
        else if (gnode->get_op_type() == "Recursion")
        {
            auto op = static_pointer_cast<Recursion>(gnode->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(op);
            auto body = op->get_body_graph();
            auto body_tu = CudaEngine().convert_graph_to_program(body, false);
            op->set_body_tu(body_tu);
        }
    }
    return true;
}
