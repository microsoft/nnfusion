// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "blockfusion/blockfusion.hpp"
#include "graph_pass_base.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/blockfusion_fused.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/util/util.hpp"

extern int BLOCKFUSION_NUM_KERNELS;
extern int BLOCKFUSION_BE_SIZE;

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class BlockFusionPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace graph
    }     // namespace pass
} // namespace nnfusion