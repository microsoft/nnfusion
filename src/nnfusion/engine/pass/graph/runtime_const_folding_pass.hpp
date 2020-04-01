// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph;

DEFINE_string(fconst_folding_backend,
              "",
              "Choose which backend will be used in Constant folding pass. Disable when not set.");

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class RuntimeConstantFoldingPass : public GraphPassBase
            {
                int runtime_const_folding_iterate_once(
                    std::shared_ptr<Graph>& graph,
                    std::set<std::shared_ptr<GNode>>& blocklist_nodes)
                {
                    int folding_cnt = 0;
                    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
                    std::set<std::shared_ptr<GNode>> const_nodes = {};
                    std::set<std::shared_ptr<GNode>> down_streams = {};

                    // Find nodes with all constant upstream nodes
                    for (auto& it : nodes)
                    {
                        if (it->is_constant())
                        {
                            const_nodes.insert(it);
                            for (auto& edge : it->get_out_edges())
                            {
                                if (edge->is_control_edge())
                                    continue;

                                CHECK(edge->get_src() == it);
                                auto dst = edge->get_dst();
                                if (blocklist_nodes.count(dst))
                                    continue;
                                if (down_streams.count(dst))
                                    continue;

                                bool inferable = true;
                                for (auto& in_edge : dst->get_in_edges())
                                {
                                    CHECK(in_edge->get_dst() == dst);
                                    auto p_const = std::dynamic_pointer_cast<op::Constant>(
                                        in_edge->get_src()->get_op_ptr());
                                    if (!in_edge->get_src()->is_constant() ||
                                        p_const->is_parameter())
                                    {
                                        inferable = false;
                                        break;
                                    }
                                }
                                if (inferable)
                                    down_streams.insert(dst);
                            }
                        }
                    }

                    for (auto& it : down_streams)
                    {
                        LOG(INFO) << ">> Found constant downstream node: " << it->get_name()
                                  << ", Op Type = " << it->get_op_type();

                        bool const_infer_success = false;
                        std::vector<std::vector<char>> raw_inputs, raw_outputs;

                        // Prepare constant inputs from upstream_nodes
                        std::set<std::shared_ptr<GNode>> upstream_nodes;
                        for (auto& input : it->get_in_edges())
                        {
                            if (input->is_control_edge())
                                continue;
                            auto const_node = input->get_src();
                            LOG(INFO)
                                << "  Input of constant downstream node: " << const_node->get_name()
                                << ", Op Type = " << const_node->get_op_type() << "/"
                                << const_node->get_op_type();

                            CHECK(input->get_dst() == it);
                            CHECK(const_node->is_constant());
                            upstream_nodes.insert(const_node);

                            auto p_const =
                                std::dynamic_pointer_cast<op::Constant>(const_node->get_op_ptr());
                            CHECK(p_const != nullptr);
                            const void* ptr = p_const->get_data_ptr();
                            size_t length = p_const->get_data_size();
                            LOG(INFO) << "  With Constant Input Node: " << p_const->get_name()
                                      << ", Memory Length = " << length;

                            std::vector<char> raw_input(length);
                            memcpy(raw_input.data(), ptr, length);
                            raw_inputs.emplace_back(std::move(raw_input));
                            CHECK(raw_input.size() == 0);
                        }

                        // Prepare runtime backend
                        nnfusion::profiler::IProfilingRuntime::Pointer runtime = nullptr;
                        std::vector<shared_ptr<const KernelRegistration>> kernel_regs;

                        if (backend == "ROCm")
                        {
                            runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
                            CHECK(runtime->check_env());
                            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                it->get_op_type(), ROCM_GPU, DT_FLOAT);
                            if (kernel_regs.size() == 0)
                                kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                    it->get_op_type(), CUDA_GPU, DT_FLOAT);
                        }
                        else if (backend == "CUDA")
                        {
                            runtime = nnfusion::profiler::CudaDefaultRuntime::Runtime();
                            CHECK(runtime->check_env());
                            kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                                it->get_op_type(), CUDA_GPU, DT_FLOAT);
                        }
                        else if (backend == "CPU")
                        {
                            runtime = nnfusion::profiler::ReferenceRuntime::Runtime();
                            CHECK(runtime->check_env());
                            // TODO: need to fill correct kernel_regs list for CPU
                            CHECK_FAIL();
                        }
                        else
                        {
                            CHECK_FAIL() << "Cannot Recognize Backend Type: " << backend;
                        }

                        // Runtime node output inference
                        shared_ptr<KernelContext> ctx(new KernelContext(it));
                        for (auto& kernel_reg : kernel_regs)
                        {
                            auto kernel = kernel_reg->m_factory(ctx);
                            if (!kernel->get_or_emit_source())
                                continue;

                            nnfusion::profiler::ProfilingContext::Pointer pctx =
                                make_shared<nnfusion::profiler::ProfilingContext>(kernel, false);

                            nnfusion::profiler::Profiler prof(runtime, pctx);
                            if (!prof.mixed_type_execute(raw_inputs, raw_outputs))
                                continue;

                            LOG(INFO) << "  For node `" << it->get_name()
                                      << "`: get runtime output results of size "
                                      << raw_outputs.size();
                            const_infer_success = true;
                            break;
                        }
                        if (!const_infer_success)
                        {
                            LOG(INFO) << "  For node `" << it->get_name()
                                      << "`: Cannot infer outputs, going to blacklist this node.";
                            blocklist_nodes.insert(it);
                            continue;
                        }

                        // Only support single output; Multi-outputs lacks output-index properties in GNode.
                        CHECK(raw_outputs.size() == 1);
#if 0 // For Debug only
						LOG(INFO) << "inputs = ";
						for (int i = 0; i < std::min(raw_inputs[0].size() / 4, 10LU); ++i)
							LOG(INFO) << (float*)raw_inputs[0].data())[i];
						puts("..");

						LOG(INFO) << "outputs = ";
						for (int i = 0; i < std::min(raw_outputs[0].size() / 4, 10LU); ++i)
							LOG(INFO) << (float*)raw_outputs[0].data())[i];
						puts("..");
#endif
                        // Ensure output layout is as expected, replace node with new_constant in place
                        CHECK(raw_outputs.size() == it->get_output_size());
                        for (int i = 0; i < it->get_output_size(); ++i)
                        {
                            auto& shape = it->get_output_shape(i);
                            auto& dtype = it->get_output_element_type(i);
                            size_t memory = dtype.size();
                            for (auto& it : shape)
                                memory *= it;
                            CHECK(memory == raw_outputs[i].size());

                            // 1. create new constant node
                            std::shared_ptr<op::Constant> new_constant_op;
                            new_constant_op =
                                std::make_shared<op::Constant>(dtype, shape, raw_outputs[i].data());
                            //new_constant_op->set_name("Constant_" + it->get_name()); // not working?
                            auto new_constant_gnode = std::make_shared<nnfusion::graph::GNode>(
                                new_constant_op, GNodeVector());

                            graph->replace_node(it, new_constant_gnode, false);

                            // remove upstream nodes with 0 out-degree
                            for (auto& node : upstream_nodes)
                            {
                                if (node->get_out_edges().size() == 0)
                                {
                                    graph->remove_node(node);
                                }
                            }

                            ++folding_cnt;
                            LOG(INFO) << "  Finish folding " << folding_cnt
                                      << "th node: name = " << it->get_unique_name() << "/"
                                      << it->get_name() << ", type = " << it->get_op_type();
                            LOG(INFO) << "";
                        }
                    }
                    return folding_cnt;
                }

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    this->backend = FLAGS_fconst_folding_backend;
                    if (this->backend == "")
                        return true;

                    static bool has_warning = false;
                    if (!has_warning)
                    {
                        has_warning = true;
                    }

                    LOG(INFO) << "Runtime Constant Folding Pass starts up for Graph: "
                              << graph->get_name();

                    // Folding output nodes results in kernel_emitter crashes
                    std::set<std::shared_ptr<GNode>> blocklist_nodes = {};
                    for (auto& node : graph->get_outputs())
                        blocklist_nodes.insert(node);

                    int folding_cnt;
                    do
                    {
                        folding_cnt = runtime_const_folding_iterate_once(graph, blocklist_nodes);
                        LOG(INFO) << ">> Runtime One Iteration Folds Infer-able Node Count: "
                                  << folding_cnt;
                    } while (folding_cnt > 0);
                    LOG(INFO) << "";
                    LOG(INFO) << ">> Runtime Constant Folding Pass ends for Graph: "
                              << graph->get_name();
                    LOG(INFO) << "";
                    return true;
                }

            private:
                std::string backend;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
