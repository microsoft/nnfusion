// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "control_flow_pass.hpp"

#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
#include "nnfusion/engine/device/cuda.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/recursion.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::engine;
using Loop = nnfusion::op::Loop;
using If = nnfusion::op::If;
using Recursion = nnfusion::op::Recursion;

void update_callers(std::shared_ptr<nnfusion::graph::Graph> graph, nnfusion::ir::Program& prog, size_t idx, std::string const_op_name) {
    std::shared_ptr<graph::GNode> const_node = nullptr;
    for (auto blk : prog)
    {
        for (auto ins : *blk)
        {
            auto node = ins->getGNode();
            if (node->get_op_ptr()->get_name() == const_op_name) {
                const_node = node;
            } else if (node->get_op_type() == "FuncForward") {
                NNFUSION_LOG(INFO) << "find caller: " << *node << "const: " << const_op_name;
                NNFUSION_CHECK(const_node != nullptr);
                NNFUSION_CHECK(node->get_input_size() == idx);
                node->set_input(idx, make_shared<nnfusion::graph::Input>(const_node->get_element_type(), const_node->get_shape()));
                graph->add_edge(const_node, 0, node, idx);
                NNFUSION_LOG(INFO) << "new node:" << *node;
                int demangle_status;
                auto func_forward_kernel = dynamic_pointer_cast<nnfusion::kernels::cuda::FuncForward>(ins->getKernel());
                func_forward_kernel->update_context_from_gnode(node);
                NNFUSION_LOG(INFO) << "forward kernel type: " << abi::__cxa_demangle(typeid(*(ins->getKernel())).name(), 0, 0, &demangle_status);
            } else if (node->get_op_type() == "Loop") {
                auto op = static_pointer_cast<Loop>(node->get_op_ptr());
                update_callers(op->get_loop_body_graph(), op->get_loop_body_tu()->program, idx, const_op_name);
            } else if (node->get_op_type() == "If") {
                auto op = static_pointer_cast<If>(node->get_op_ptr());
                update_callers(op->get_then_branch_graph(), op->get_then_branch_tu()->program, idx, const_op_name);
                update_callers(op->get_else_branch_graph(), op->get_else_branch_tu()->program, idx, const_op_name);
            } else if (node->get_op_type() == "Recursion") {
                auto op = static_pointer_cast<Recursion>(node->get_op_ptr());
                update_callers(op->get_body_graph(), op->get_body_tu()->program, idx, const_op_name);
            }
        }
    }
}

// this func append Constants in the subgraph as additional inputs to the control flow node
void extract_constant_nodes(std::shared_ptr<nnfusion::graph::Graph>& graph,
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
                auto recursive_op = std::dynamic_pointer_cast<op::Recursion>(gnode->get_op_ptr());
                if (recursive_op != nullptr) {
                    update_callers(recursive_op->get_body_graph(), recursive_op->get_body_tu()->program, idx, node->get_op_ptr()->get_name());
                }
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
            extract_constant_nodes(graph, gnode, body_tu->program);
        }
    }
    return true;
}
