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

#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

REGISTER_BACKWARD_TRANSLATOR(DropoutTraining)
    .translator([](std::shared_ptr<GNode> forward_node,
                   const GNodeIndexVector& outputs_grad,
                   std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        // y, y_mask = dropout(x)

        auto y_grad_index = outputs_grad.at(0);
        auto mask_index = get_node_output(forward_node, 1);

        auto dropout_op = std::dynamic_pointer_cast<op::GenericOp>(forward_node->get_op_ptr());
        nnfusion::op::OpConfig::any myConfig;
        myConfig["ratio"] = dropout_op->localOpConfig.getRoot()["ratio"];

        auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_grad", "DropoutTrainingGrad", myConfig);
        auto generic_gnode = graph->add_node_and_edge(generic_op, {y_grad_index, mask_index});

        return GNodeIndexVector{GNodeIndex{generic_gnode, 0}};
    });