// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class ProfilingBasedKernelSelector : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

                pair<NNFusion_DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
                    profiling_best(shared_ptr<GNode> gnode,
                                   NNFusion_DeviceType devtype,
                                   nnfusion::profiler::IProfilingRuntime::Pointer runtime);
            };

            class DefaultKernelSelector : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
                pair<NNFusion_DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
                    pick_first(shared_ptr<GNode> gnode, NNFusion_DeviceType devtype);
                bool register_custom_kernel(std::string op, NNFusion_DeviceType devtype);
            };

            class FetchBasedSelector : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
                pair<NNFusion_DeviceType, nnfusion::kernels::KernelEmitter::Pointer>
                    fetch_inventory(shared_ptr<cache::KernelCacheManager> cache_manager,
                                    shared_ptr<GNode> gnode,
                                    NNFusion_DeviceType devtype);
            };
        }
    }
} // namespace nnfusion