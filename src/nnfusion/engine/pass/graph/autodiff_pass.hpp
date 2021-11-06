//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/pass/graph/graph_pass_base.hpp"

DECLARE_bool(fautodiff);
DECLARE_string(ftraining_optimizer);

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AutodiffPass : public GraphPassBase
            {
            public:
                bool run_on_graph(
                    std::shared_ptr<nnfusion::graph::Graph>& graph,
                    std::shared_ptr<vector<vector<float>>> backward_inputs);
            };
        }
    }
}
