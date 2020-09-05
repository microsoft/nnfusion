// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            // Address the perf gap when dot transB is on/off.
            // If a dot kernel is faster with transB=true, the transpose op,
            // which is fused by the end, is inserted before current dot.
            //      const                 const
            //       / \                    |
            //   dot_a  dot_b     ->      trans
            // (transB = false)            / \
            //                        dot_a  dot_b
            //                       (transB = true)
            class DotTransposePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
