// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_multi_codegen_pass.hpp"
#include <regex>
#include "codegen_langunit.hpp"
#include "codegenerator_helper.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/cpu/barrier.hpp"
#include "nnfusion/core/kernels/cpu/cpu_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

DEFINE_bool(fmulti_shape, false, "Enable multiple input shape mode for ONNX.");

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;
using namespace nnfusion::codegen;
using namespace nnfusion::async;

bool CudaMultiCodegenPassPre::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    initialize(ctx, tu);
    NNFUSION_CHECK(collect_mem(ctx, tu));
    NNFUSION_CHECK(collect_stream(ctx, tu));
    NNFUSION_CHECK(collect_funcs(ctx, tu));
    NNFUSION_CHECK(modify_codegen());
    NNFUSION_LOG(INFO) << "Codegen for " << get_device_str(device_type()) << " done.";
    return true;
}