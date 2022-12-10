// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rocm.hpp"
#include "reversed_dfs_visitor.hpp"

#include "nnfusion/engine/pass/graph/assign_async_info_pass.hpp"
#include "nnfusion/engine/pass/graph/assign_layout_pass.hpp"
#include "nnfusion/engine/pass/graph/autodiff_pass.hpp"
#include "nnfusion/engine/pass/graph/batchnorm_inference_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/blockfusion_pass.hpp"
#include "nnfusion/engine/pass/graph/common_subexpression_elimination_pass.hpp"
#include "nnfusion/engine/pass/graph/dot_transpose_pass.hpp"
#include "nnfusion/engine/pass/graph/gemm_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/gnode_device_dispatcher.hpp"
#include "nnfusion/engine/pass/graph/gradient_weight_mapping_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_profiling_pass.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/pass/graph/kernel_tuning.hpp"
#include "nnfusion/engine/pass/graph/multi_reshape_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/pass/graph/pattern_substitution.hpp"
#include "nnfusion/engine/pass/graph/reduce_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/register_fusion_pass.hpp"
#include "nnfusion/engine/pass/graph/runtime_const_folding_pass.hpp"
#include "nnfusion/engine/pass/graph/split_softmax_pass.hpp"
#include "nnfusion/engine/pass/graph/vector_dot_transpose_pass.hpp"

#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/tensor/inplace_tensor_analysis.hpp"
#include "nnfusion/engine/pass/tensor/liveness_analysis.hpp"
#include "nnfusion/engine/pass/tensor/tensor_device_dispatcher.hpp"
#include "nnfusion/engine/pass/tensor/tensor_memory_layout.hpp"

#include "nnfusion/engine/pass/codegen/rocm_codegen_pass.hpp"

using namespace nnfusion;
using namespace nnfusion::engine;
using namespace nnfusion::pass::graph;
using namespace nnfusion::pass;

ROCmEngine::ROCmEngine()
    : Engine()
{
    g_passes->push_back(make_shared<CSEPass>());
    g_passes->push_back(make_shared<AutodiffPass>());
    g_passes->push_back(make_shared<GradientWeightMappingPass>());
    g_passes->push_back(make_shared<RuntimeConstantFoldingPass>());
    g_passes->push_back(make_shared<MultiReshapeFoldingPass>());
    g_passes->push_back(make_shared<VectorDotTransposePass>());
    g_passes->push_back(make_shared<GemmFusionPass>());
    g_passes->push_back(make_shared<BatchNormInferenceFoldingPass>());
    g_passes->push_back(make_shared<AssignLayoutPass>());
    g_passes->push_back(make_shared<OpInplacePass>());
    g_passes->push_back(make_shared<ReduceFusionPass>());

    g_passes->push_back(make_shared<PatternSubstitutionPass>());
    g_passes->push_back(make_shared<SplitSoftmaxPass>());
    g_passes->push_back(make_shared<RegisterFusionPass>());
    // Kernel selection
    g_passes->push_back(make_shared<DefaultGNodeDeviceDispatcher>());
    g_passes->push_back(make_shared<KernelFusionPass>());
    g_passes->push_back(make_shared<KernelTuning>());
    g_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    g_passes->push_back(make_shared<FetchBasedSelector>());
    g_passes->push_back(make_shared<DefaultKernelSelector>());

    // GPU specific graph passes
    g_passes->push_back(make_shared<KernelProfilingPass>());
    g_passes->push_back(make_shared<PatternSubstitutionPass>());
    g_passes->push_back(make_shared<BlockFusionPass>());

    // Specific opt for dot
    g_passes->push_back(make_shared<DotTransposePass>());

    // Assign stream passes
    g_passes->push_back(make_shared<AssignAsyncInfoPass>());

    // Visitor
    g_visitor = make_shared<ReversedDFSVisitor>();

    // extract graph signature
    m_passes->push_back(make_shared<ExtractGraphSignature>());
    // Do tensor allocation plan
    m_passes->push_back(make_shared<TensorDeviceDispatcher>());
    m_passes->push_back(make_shared<TensorLivenessAnalysis>());
    m_passes->push_back(make_shared<InplaceTensorAnalysis>());
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

    // Do codegen
    m_passes->push_back(make_shared<RocmCodegenPass>());
}
