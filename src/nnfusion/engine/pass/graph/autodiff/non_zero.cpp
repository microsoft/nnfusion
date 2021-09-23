// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(NonZero).translator(
    [](std::shared_ptr<GNode> forward_node,
       const GNodeIndexVector& outputs_grad,
       std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        return GNodeIndexVector{pass::graph::autodiff::DiffEngine::EMPTY_GNODE_INDEX};
    });