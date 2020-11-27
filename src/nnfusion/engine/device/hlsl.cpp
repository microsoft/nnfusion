// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl.hpp"
#include "degree_based_visitor.hpp"
#include "nnfusion/engine/pass/codegen/hlsl_codegen_pass.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/reduce_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::engine;

HLSLEngine::HLSLEngine()
    : Engine()
{
    g_passes->push_back(make_shared<GradientWeightMappingPass>());
    g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
    g_passes->push_back(make_shared<ReduceFusionPass>());

    // Kernel selection
    g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
    g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    g_passes->push_back(make_shared<FetchBasedSelector>());
    g_passes->push_back(make_shared<DefaultKernelSelector>());

    // Visitor
    g_visitor = make_shared<DegreeBasedVisitor>();

    // Do codegen
    m_passes->push_back(make_shared<HLSLCodegenPass>());
}
