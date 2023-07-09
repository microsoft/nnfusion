// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gnode_device_dispatcher.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DEFINE_string(fdefault_device,
              "CUDA",
              "Choose defualt device from [CUDA, CPU, ROCm] in the codegen.");
DECLARE_bool(frt_const_folding);
DEFINE_int32(fnum_non_cpu, 1, "Number of gpus.");
bool DefaultGNodeDeviceDispatcher::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    NNFusion_DeviceType dt = nnfusion::get_device_type(dev_name);
    int num_device = FLAGS_fnum_non_cpu;
    /* for debug purpose
    switch(default_device)
    {
        case GENERIC_CPU:
        LOG(INFO) << "GENERIC_CPU";
        break;
        case  ROCM_GPU:
        LOG(INFO) << "ROCM_GPU";
        break;
        case CUDA_GPU:
        LOG(INFO) << "CUDA_GPU";
    }
    */

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        it->Set<NNFusion_DeviceType>("DeviceType", move(dt));
        it->Set<int>("DeviceID", 0);
    }

    int count = 0;
    const GNodeVector& start = graph->get_outputs();
    // Stack of work to do.
    std::vector<std::shared_ptr<GNode>> stack(start.size());

    for (int i = 0; i < start.size(); ++i)
    {
        stack[i] = start[i];
    }

    std::vector<bool> visited(graph->get_max_node_id(), false);
    while (!stack.empty())
    {
        std::shared_ptr<GNode> gnode = stack.back();
        stack.pop_back();
        if (visited[gnode->get_id()])
        {
            continue;
        }
        visited[gnode->get_id()] = true;
        gnode->Set<int>("DeviceID", move(count));
        auto add_gnode = [&visited, &stack](std::shared_ptr<GNode> in_node) {
            if (!visited[in_node->get_id()])
            {
                // Note; we must not mark as visited until we actually process it.
                stack.push_back(in_node);
            }
        };

        size_t pre = stack.size();

        for (auto in_edge : gnode->get_in_edges())
        {
            add_gnode(in_edge->get_src());
        }

        if (stack.size() == pre)
        {
            if (count < num_device - 1)
                count += 1;
        }
    }

    for (auto gnode : nodes)
    {
        // all constant ops use default stream
        if (gnode->get_op_type() == "Constant" || gnode->get_op_ptr()->is_parameter())
        {
            gnode->Set<int>("DeviceID", 0);
        }
    }
    return true;
}