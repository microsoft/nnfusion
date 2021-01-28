// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "assign_async_info_pass.hpp"
#include "assign_layout_pass.hpp"
#include "autodiff_pass.hpp"
#include "batchnorm_inference_folding_pass.hpp"
#include "blockfusion_pass.hpp"
#include "codegen_graphcore_pass.hpp"
#include "common_subexpression_elimination_pass.hpp"
#include "dot_transpose_pass.hpp"
#include "gemm_fusion_pass.hpp"
#include "gnode_device_dispatcher.hpp"
#include "gradient_weight_mapping_pass.hpp"
#include "kernel_fusion_pass.hpp"
#include "kernel_profiling_pass.hpp"
#include "kernel_selection.hpp"
#include "multi_reshape_folding_pass.hpp"
#include "op_inplace_pass.hpp"
#include "pattern_substitution.hpp"
#include "reduce_fusion_pass.hpp"
#include "runtime_const_folding_pass.hpp"
#include "vector_dot_transpose_pass.hpp"
using namespace nnfusion::pass::graph;
using namespace std;

DEFINE_bool(ffold_reshape_op, true, "Folding Reshape operators.");
DEFINE_bool(ftranspose_vecdot, false, "Enable vectdot transpose.");
DEFINE_string(fantares_codegen_server,
              "",
              "Antares codegen server address and port, format: <ip>:<port>");

DECLARE_string(fdefault_device);
