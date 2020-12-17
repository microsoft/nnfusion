// Microsoft (c) 2019, NNFusion Team

#include "graph_pass.hpp"
#include "manager.hpp"

#include "assign_async_info_pass.hpp"
#include "assign_layout_pass.hpp"
#include "autodiff_pass.hpp"
#include "batchnorm_inference_folding_pass.hpp"
#include "blockfusion_pass.hpp"
#include "codegen_dxcompute_pass.hpp"
#include "codegen_graphcore_pass.hpp"
#include "common_subexpression_elimination_pass.hpp"
#include "dot_transpose_pass.hpp"
#include "gemm_fusion_pass.hpp"
#include "gnode_device_dispatcher.hpp"
#include "gradient_weight_mapping_pass.hpp"
#include "graph_serialization_pass.hpp"
#include "kernel_fusion_pass.hpp"
#include "kernel_profiling_pass.hpp"
#include "kernel_selection.hpp"
#include "multi_reshape_folding_pass.hpp"
#include "op_inplace_pass.hpp"
#include "pattern_substitution.hpp"
#include "runtime_const_folding_pass.hpp"
#include "superscaler_dataparallelism_pass.hpp"
#include "vector_dot_transpose_pass.hpp"

using namespace nnfusion::pass::graph;
using namespace std;

DEFINE_bool(ffold_reshape_op, true, "Folding Reshape operators.");
DEFINE_bool(ftranspose_vecdot, false, "Enable vectdot transpose.");
DEFINE_string(fantares_codegen_server,
              "",
              "Antares codegen server address and port, format: <ip>:<port>");

DECLARE_string(fdefault_device);

bool GraphPass::run(std::vector<std::shared_ptr<Graph>>& graph_vec)
{
    GraphPassManager pass_manager;
    // Generate result op must before LivenessPass
    // Generate result is implemented in gradient weight mapping pass
    pass_manager.register_pass<CSEPass>();
    pass_manager.register_pass<AutodiffPass>();
    pass_manager.register_pass<GradientWeightMappingPass>();
    pass_manager.register_pass<RuntimeConstantFoldingPass>();
    pass_manager.register_pass<MultiReshapeFoldingPass>();
    pass_manager.register_pass<VectorDotTransposePass>();
    pass_manager.register_pass<GemmFusionPass>();
    pass_manager.register_pass<BatchNormInferenceFoldingPass>();
    pass_manager.register_pass<AssignLayoutPass>();
    //superscaler pass
    pass_manager.register_pass<SuperScalerDataParallelismPass>();
    pass_manager.register_pass<GraphSerializationPass>();

    pass_manager.register_pass<OpInplacePass>();

    pass_manager.register_pass<PatternSubstitutionPass>();
    // The graph after this pass will have selected kernels
    pass_manager.register_pass<DefaultGNodeDeviceDispatcher>();
    pass_manager.register_pass<ProfilingBasedKernelSelector>();
    pass_manager.register_pass<FetchBasedSelector>();
    pass_manager.register_pass<DefaultKernelSelector>();
    pass_manager.register_pass<AntaresProfilingBasedKernelSelector>();

    // Specific opt for dot
    pass_manager.register_pass<DotTransposePass>();

    // GPU specific graph passes
    pass_manager.register_pass<KernelFusionPass>();
    pass_manager.register_pass<KernelProfilingPass>();
    pass_manager.register_pass<PatternSubstitutionPass>();
    pass_manager.register_pass<BlockFusionPass>();

    // assign stream
    pass_manager.register_pass<AssignAsyncInfoPass>();

    return pass_manager.run_passes(graph_vec);
}
