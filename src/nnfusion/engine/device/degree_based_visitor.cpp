// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "degree_based_visitor.hpp"
#include "nnfusion/common/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;

nnfusion::ir::Program::Pointer DegreeBasedVisitor::run_on_graph(shared_ptr<graph::Graph> graph,
                                                                EngineContext::Pointer context)
{
    NNFUSION_LOG(INFO) << "Translating graph:\t" << graph->get_name();

    auto program =
        make_shared<ir::Program>(nnfusion::ir::Program::create_single_basic_block_program());
    auto bb_main = program->get_entry();

    // Translate the Node
    // Currently:
    // * Translate each gnode into an instruction;
    // * Store all instruction inside one basicblock since we don't have
    //   control-flow by now.
    GNodeVector node_vec;
    auto nodes = graph->get_nodes();
    std::unordered_map<std::shared_ptr<GNode>, int> din, dout;
    std::unordered_set<std::shared_ptr<GNode>> visited, vis_pend;

    // Count degrees
    for (auto& it : nodes)
    {
        for (auto& in_edge : it->get_in_edges())
        {
            if (in_edge->is_control_edge())
                continue;
            NNFUSION_CHECK(in_edge->get_dst() == it);
            din[it]++;
            dout[in_edge->get_src()]++;
        }
    }

    // legality checks
    for (auto& it : nodes)
    {
        NNFUSION_CHECK(it.get() != nullptr);
        if (din[it] == 0 && dout[it] == 0)
            visited.insert(it), context->blacklist.insert(it);
        NNFUSION_CHECK(it->get_output_size() == 1);
    }
    NNFUSION_LOG(INFO) << "There are " << context->blacklist.size()
                       << " standalone GNode(s) found.";

    // Fill offsetup nodes
    std::deque<std::shared_ptr<GNode>> gen_q, pend_q;
    for (auto& it : nodes)
    {
        if (visited.count(it))
            continue;
        if (din[it] == 0)
        {
            gen_q.push_back(it);
        }
    }

    // Perform blockfusion
    int offset = 0, step = 0;
    auto new_super_step = [&]() {
        while (pend_q.size())
        {
            gen_q.push_back(pend_q.front());
            pend_q.pop_front();
        }
        if (offset > 0)
            ++step, offset = 0;
    };

    while (gen_q.size() > 0 || pend_q.size() > 0)
    {
        // Move to new super step if satisifed
        if (!gen_q.size())
            new_super_step();

        auto curr = gen_q.front();
        gen_q.pop_front();
        visited.insert(curr);

        node_vec.push_back(curr);

        // Check its children about whether all inputs are ready (Must be put after any possible new_super_step())
        for (auto& edge : curr->get_out_edges())
        {
            if (edge->is_control_edge())
                continue;
            NNFUSION_CHECK(edge->get_src() == curr);
            NNFUSION_CHECK(visited.count(edge->get_dst()) == 0);

            bool ready = true;
            for (auto& from : edge->get_dst()->get_in_edges())
            {
                if (from->is_control_edge())
                    continue;
                if (visited.count(from->get_src()) == 0)
                {
                    ready = false;
                    break;
                }
            }
            if (ready)
            {
                // Only join pend_q once
                if (vis_pend.count(edge->get_dst()) == 0)
                {
                    vis_pend.insert(edge->get_dst());
                    pend_q.push_back(edge->get_dst());
                }
            }
        }
    }

    for (auto gnode : node_vec)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
        ir->setGNode(gnode);
        ir->copy_tags_from(*gnode);
        ir->setName(gnode->get_name());
        bb_main->push_back(ir);
    }

    return program;
}