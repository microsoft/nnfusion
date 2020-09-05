// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "graphcore.hpp"
#include "nnfusion/engine/pass/graph/codegen_graphcore_pass.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::engine;

GraphCoreEngine::GraphCoreEngine()
    : Engine()
{
    g_passes->push_back(make_shared<GradientWeightMappingPass>());
    g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
    g_passes->push_back(make_shared<GraphCoreCodegenPass>());

    g_visitor = nullptr;
    m_passes = nullptr;
}
