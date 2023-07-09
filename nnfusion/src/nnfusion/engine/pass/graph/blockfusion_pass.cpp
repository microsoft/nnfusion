// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "blockfusion_pass.hpp"
#include "blockfusion/blockfusion.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;

// It's not encouraging to use the blockfusion level 2 because it is still an experimental feature and is under development
DEFINE_int32(
    fblockfusion_level,
    1,
    "BlockFusion optimization level: 0: disable, 1: wavefront, 2: wavefront with wave merge");
DEFINE_bool(fblockfusion_interplay,
            true,
            "Interplay of intra- and inter- operator scheduling in BlockFusion");
DEFINE_bool(fblockfusion_check_correctness,
            false,
            "Check the correctness of BlockFusion codegen and fallback to original execution when "
            "failure detected");
DECLARE_string(fproduct_name);
DECLARE_string(fdefault_device);

bool BlockFusionPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    if (FLAGS_fblockfusion_level == 0)
    {
        NNFUSION_LOG(INFO) << "BlockFusion is disabled.";
        return true;
    }

    std::shared_ptr<BlockFusionOptimizer> optimizer;
    if (FLAGS_fblockfusion_level == 1)
    {
        optimizer =
            std::make_shared<BlockFusionWavefrontOptimizer>(graph,
                                                            FLAGS_fdefault_device,
                                                            FLAGS_fproduct_name,
                                                            1,
                                                            FLAGS_fblockfusion_interplay,
                                                            FLAGS_fblockfusion_check_correctness);
    }
    else if (FLAGS_fblockfusion_level == 2)
    {
        optimizer =
            std::make_shared<BlockFusionWavefrontOptimizer>(graph,
                                                            FLAGS_fdefault_device,
                                                            FLAGS_fproduct_name,
                                                            2,
                                                            FLAGS_fblockfusion_interplay,
                                                            FLAGS_fblockfusion_check_correctness);
    }

    if (optimizer->Optimize())
    {
        return true;
    }
    else
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "BlockFusion pass failed, fallback.";
        return true;
    }
    return true;
}
