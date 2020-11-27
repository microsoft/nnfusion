// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Unittest for profiler feature of engine
 * \author wenxh
 */

#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::profiler;

TEST(nnfusion_engine_profiler, basic_utils)
{
    // Prepare
    auto gnode = inventory::create_object<op::Pad, float>(0);

    EXPECT_TRUE(gnode != nullptr);

    // Filter out the kernels meeting the requirement;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), CUDA_GPU, element::f32);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    // Gnerate Test data
    auto input = nnfusion::inventory::generate_input<op::Pad, float>(0);
    auto output = nnfusion::inventory::generate_output<op::Pad, float>(0);
    vector<vector<float>> inputs;
    inputs.push_back(vector<float>{/*a*/ 1, 2, 3, 4, 5, 6});
    inputs.push_back(vector<float>{/*b*/ 2112});
    vector<vector<float>> outputs;
    outputs.push_back(output);

    EXPECT_GT(kernel_regs.size(), 0);
    bool has_valid_kernel = false;
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            ProfilingContext::Pointer pctx = make_shared<ProfilingContext>(kernel);
            Profiler ref_prof(ReferenceRuntime::Runtime(), pctx);
            auto res = ref_prof.execute();
            EXPECT_TRUE(res);
            NNFUSION_LOG(INFO)
                << "Profiling of Pad operator using Reference: Avg Host duration(ms) "
                << pctx->result.get_host_avg();
            pctx->reset();

            auto res_val = ref_prof.execute(inputs);
            EXPECT_EQ(res_val.size(), outputs.size());
            for (int i = 0; i < res_val.size(); i++)
                EXPECT_TRUE(test::all_close_f(res_val[i], outputs[i]));
            NNFUSION_LOG(INFO) << "Profiling of Pad operator has correct result.";
        }
    }
    EXPECT_TRUE(has_valid_kernel);
}