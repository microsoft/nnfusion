// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/util/annotations.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class OpInplacePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

                // private:
                //     template <class T>
                //     void AddInplace(T op, size_t output, size_t input, bool destructive)
                //     {
                //         auto op_annotations = op->get_op_annotations();
                //         if (op_annotations)
                //         {
                //             // pass-through
                //             op_annotations->add_in_place_oi_pair({output, input, destructive});
                //         }
                //         else
                //         {
                //             op_annotations = std::make_shared<Annotations>();
                //             // pass-through
                //             op_annotations->add_in_place_oi_pair({output, input, destructive});
                //             op->set_op_annotations(op_annotations);
                //         }
                //     }
            };
        }
    }
}
