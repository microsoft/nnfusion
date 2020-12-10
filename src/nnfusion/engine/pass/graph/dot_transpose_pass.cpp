// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dot_transpose_pass.hpp"
#include "gnode_device_dispatcher.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "runtime_const_folding_pass.hpp"

DEFINE_bool(fdot_transpose, false, "Dot transpose.");
// official product name for cuda: > nvidia-smi -x -q | grep product_name | sed -n '1p' | cut -d \> -f 2 | cut -d \< -f 1
DECLARE_string(fproduct_name);

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

namespace
{
    using nnfusion::cache::KernelEntry;
    nnfusion::cache::KernelEntry fetch_best_kernel(
        const string& identifier,
        const string& platform,
        const set<string>& tags, // should exactly match every tag, no more no less
        std::shared_ptr<nnfusion::cache::KernelCacheManager> cache_manager,
        const string& product_name)
    {
        auto fetched = cache_manager->fetch_all(identifier, platform);
        std::vector<KernelEntry> matched_kernels;
        KernelEntry best_kernel;
        for (auto matched_kernel_p : fetched)
        {
            auto matched_kernel = *matched_kernel_p;
            if (matched_kernel.tags == tags)
            {
                bool is_better = false;
                if (best_kernel.function.is_null())
                {
                    is_better = true;
                }
                else if (matched_kernel.profile.find(product_name) != matched_kernel.profile.end())
                {
                    if (best_kernel.profile.find(product_name) == best_kernel.profile.end() ||
                        best_kernel.profile.at(product_name) >
                            matched_kernel.profile.at(product_name))
                    {
                        is_better = true;
                    }
                }
                if (is_better)
                {
                    best_kernel = matched_kernel;
                }
            }
        }
        return best_kernel;
    }

    kernels::KernelEmitter::Pointer generate_func_point(shared_ptr<GNode> gnode,
                                                        nnfusion::cache::KernelEntry_p kernel_entry)
    {
        shared_ptr<KernelContext> ctx(new KernelContext(gnode));
        if (kernel_entry != nullptr)
        {
            auto kernel = std::make_shared<kernels::cuda::CacheBlockCudaEmitter>(ctx, kernel_entry);
            if (kernel->get_or_emit_source())
            {
                return kernel;
            }
        }
        return nullptr;
    }

    string get_product_name() { return FLAGS_fproduct_name; }
}

bool DotTransposePass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool using_pass = FLAGS_fdot_transpose;
    if (!using_pass)
        return true;

    auto cache_manager = std::make_shared<cache::KernelCacheManager>();
    if (!cache_manager->is_valid())
    {
        NNFUSION_LOG(INFO) << "No valid kernel cache, ignore dot transpose pass";
        return true;
    }

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();

    for (auto& it : nodes)
    {
        if (it->get_op_type() != "Dot")
        {
            continue;
        }
        if (!(*it)["DeviceType"].is_valid() || !(*it)["DeviceID"].is_valid())
        {
            NNFUSION_LOG(NNFUSION_WARNING)
                << "GNode DeviceType and DeviceID should be assigned before this pass："
                << it->get_name();
            continue;
        }
        {
            auto dot = std::dynamic_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(dot);
            // already transposed or handled by this pass
            if (dot->get_transpose_B())
            {
                continue;
            }
        }
        auto input1_edge = it->get_in_edge(1);
        NNFUSION_CHECK(input1_edge);
        auto input1_gnode = input1_edge->get_src();
        auto input1_index = input1_edge->get_src_output();
        // input1 should be a const
        if (!input1_gnode->is_constant() || input1_gnode->get_shape().size() != 2)
        {
            continue;
        }

        ///\todo ignore weight to avoid inplace updating?
        // auto const_op =
        //     std::dynamic_pointer_cast<nnfusion::op::Constant>(input1_gnode->get_op_ptr());
        // if (const_op->is_weight())
        // {
        //     continue;
        // }

        shared_ptr<KernelContext> ctx(new KernelContext(it));
        std::string identifier = ctx->generate_identifier();
        if (identifier == "")
        {
            continue;
        }

        // check all reference to const is dot, and const is the rhs
        bool different_reference = false;
        for (auto out_edge : input1_gnode->get_output_users(input1_index))
        {
            if (out_edge->is_control_edge())
            {
                different_reference = true;
                break;
            }
            shared_ptr<KernelContext> cur_ctx(new KernelContext(out_edge->get_dst()));
            std::string cur_identifier = ctx->generate_identifier();
            if (cur_identifier != identifier)
            {
                different_reference = true;
                break;
            }
            if (out_edge->get_dst_input() != 1)
            {
                different_reference = true;
                break;
            }
        }
        if (different_reference)
        {
            continue;
        }

        auto platform = nnfusion::get_device_str((*it)["DeviceType"].as<NNFusion_DeviceType>());
        auto product_name = get_product_name();
        nnfusion::cache::KernelEntry dot_kernel =
            fetch_best_kernel(identifier, platform, set<string>{}, cache_manager, product_name);
        nnfusion::cache::KernelEntry transpose_dot_kernel = fetch_best_kernel(
            identifier, platform, set<string>{"transB"}, cache_manager, product_name);
        // no profiling time
        if (dot_kernel.profile.find(product_name) == dot_kernel.profile.end() ||
            transpose_dot_kernel.profile.find(product_name) == transpose_dot_kernel.profile.end())
        {
            continue;
        }
        NNFUSION_LOG(INFO) << "Dot time: " << dot_kernel.profile.at(product_name)
                           << ", Transpose dot time: "
                           << transpose_dot_kernel.profile.at(product_name);
        if (dot_kernel.profile.at(product_name) <= transpose_dot_kernel.profile.at(product_name))
        {
            continue;
        }

        // insert transpose
        NNFUSION_LOG(INFO) << "Transpose constant: " << input1_gnode->get_name();
        auto trans_gnode =
            nnfusion::graph::numpy_transpose(input1_gnode, nnfusion::AxisVector(), input1_index);
        graph->add_node(trans_gnode);
        graph->add_edge(input1_gnode, input1_index, trans_gnode, 0);
        // reconnect dot nodes
        for (auto out_edge : input1_gnode->get_output_users(input1_index))
        {
            auto dst_node = out_edge->get_dst();
            if (dst_node == trans_gnode)
            {
                continue;
            }
            graph->remove_edge(out_edge);
            auto new_input = make_shared<nnfusion::graph::Input>(
                dst_node->get_input_element_type(1), trans_gnode->get_shape());
            dst_node->set_input(1, new_input);
            graph->add_edge(trans_gnode, 0, dst_node, 1);
            auto dot = std::dynamic_pointer_cast<nnfusion::op::Dot>(dst_node->get_op_ptr());
            NNFUSION_CHECK(dot);
            dot->get_transpose_B() = true;

            auto func_p = generate_func_point(
                dst_node, std::make_shared<nnfusion::cache::KernelEntry>(transpose_dot_kernel));
            NNFUSION_CHECK(func_p);
            if (func_p)
                (*dst_node)["Kernel_Selection_Result"] =
                    std::make_pair((*it)["DeviceType"].as<NNFusion_DeviceType>(), func_p);
        }
    }
    // folding trans node
    // RuntimeConstantFoldingPass().run_on_graph(graph);
    // assign kernel for transpose
    DefaultGNodeDeviceDispatcher().run_on_graph(graph);
    DefaultKernelSelector().run_on_graph(graph);

    return true;
}
