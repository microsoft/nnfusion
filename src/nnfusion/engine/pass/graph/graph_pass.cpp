// Microsoft (c) 2019, NNFusion Team

#include "graph_pass.hpp"
#include "manager.hpp"

#include "assign_async_info_pass.hpp"
#include "assign_layout_pass.hpp"
#include "device_dispatcher.hpp"
#include "gemm_fusion_pass.hpp"
#include "gradient_weight_mapping_pass.hpp"
#include "kernel_fusion_pass.hpp"
#include "kernel_selection.hpp"
#include "multi_reshape_folding_pass.hpp"
#include "op_inplace_pass.hpp"
#include "runtime_const_folding_pass.hpp"
#include "vector_dot_transpose_pass.hpp"

using namespace nnfusion::pass::graph;
using namespace std;

bool GraphPass::run(std::shared_ptr<Graph> graph)
{
    GraphPassManager pass_manager;
    // Generate result op must before LivenessPass
    // Generate result is implemented in gradient weight mapping pass
    pass_manager.register_pass<GradientWeightMappingPass>();
    pass_manager.register_pass<RuntimeConstantFoldingPass>();
    pass_manager.register_pass<MultiReshapeFoldingPass>();
    pass_manager.register_pass<VectorDotTransposePass>();
    pass_manager.register_pass<GemmFusionPass>();
    pass_manager.register_pass<AssignLayoutPass>();
    pass_manager.register_pass<OpInplacePass>();

    // The graph after this pass will have selected kernels
    pass_manager.register_pass<DefaultDeviceDispatcher>();
    pass_manager.register_pass<ProfilingBasedKernelSelector>();
    pass_manager.register_pass<DefaultKernelSelector>();

    // GPU specific graph passes
    pass_manager.register_pass<KernelFusionPass>();

    // assign stream
    pass_manager.register_pass<AssignAsyncInfoPass>();

    pass_manager.run_passes(graph);

    return true;
}
