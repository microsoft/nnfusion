// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rocm.hpp"
#include "reversed_dfs_visitor.hpp"

#include "nnfusion/engine/pass/graph/assign_async_info_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_layout_pass.hpp"
#include "nnfusion/engine/pass/graph/blockfusion_pass.hpp"
#include "nnfusion/engine/pass/graph/gemm_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_profiling_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/multi_reshape_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/vector_dot_transpose_pass.hpp"

#include "nnfusion/engine/pass/codegen/rocm_codegen_pass.hpp"
#include "nnfusion/engine/pass/tensor/inplace_tensor_analysis.hpp"
#include "nnfusion/engine/pass/tensor/liveness_analysis.hpp"
#include "nnfusion/engine/pass/tensor/tensor_device_dispatcher.hpp"
#include "nnfusion/engine/pass/tensor/tensor_memory_layout.hpp"

using namespace nnfusion;
using namespace nnfusion::engine;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass;

ROCmEngine::ROCmEngine()
    : Engine()
{
    g_passes->push_back(make_shared<GradientWeightMappingPass>());
    g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
    g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
    g_passes->push_back(make_shared<VectorDotTransposePass>());
    g_passes->push_back(make_shared<GemmFusionPass>());
    g_passes->push_back(make_shared<AssignLayoutPass>());
    g_passes->push_back(make_shared<OpInplacePass>());

    // Kernel selection
    g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
    g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    g_passes->push_back(make_shared<FetchBasedSelector>());
    g_passes->push_back(make_shared<DefaultKernelSelector>());

    // GPU specific graph passes
    g_passes->push_back(make_shared<KernelFusionPass>());
    g_passes->push_back(make_shared<KernelProfilingPass>());
    g_passes->push_back(make_shared<BlockFusionPass>());

    // Assign stream passes
    g_passes->push_back(make_shared<AssignAsyncInfoPass>());

    // Visitor
    g_visitor = make_shared<ReversedDFSVisitor>();

    // Do tensor allocation plan
    m_passes->push_back(make_shared<TensorDeviceDispatcher>());
    m_passes->push_back(make_shared<TensorLivenessAnalysis>());
    m_passes->push_back(make_shared<InplaceTensorAnalysis>());
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

    // Do codegen
    m_passes->push_back(make_shared<RocmCodegenPass>());
}
