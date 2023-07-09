// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Basic Test Example for AvgPool;
 * \author wenxh
 */

#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/core/operators/op_define/pad.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

TEST(nnfusion_core_kernels, sample)
{
    // Prepare
    auto gnode = inventory::create_object<op::Pad, float>(0);
    EXPECT_TRUE(gnode != nullptr);

    // Filter out the kernels meeting the requirement;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(
            gnode->get_op_type(), CUDA_GPU, element::f32);
    shared_ptr<KernelContext> ctx(new KernelContext(gnode));

    EXPECT_GT(kernel_regs.size(), 0);
    bool has_valid_kernel = false;
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;

            NNFUSION_LOG(INFO) << "Now on Kernel Emitter:\t" << kernel->get_function_name();

            // Data
            DataBuffer output(element::f32);
            output.loadVector(nnfusion::inventory::generate_output<op::Pad, float>(0));
            vector<DataBuffer> inputs;
            inputs.push_back(DataBuffer::fromList<float>({/*a*/ 1, 2, 3, 4, 5, 6}));
            inputs.push_back(DataBuffer::fromList<float>({/*b*/ 2112}));
            vector<DataBuffer> outputs;
            outputs.push_back(output);
            // Context
            nnfusion::profiler::ProfilingContext::Pointer pctx =
                make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            auto rocm_runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
            // Rocm
            if (rocm_runtime->check_env())
            {
                NNFUSION_LOG(INFO) << "Test ROCM runtime of Pad operator:";
                nnfusion::profiler::Profiler prof(rocm_runtime, pctx);
                prof.execute();
                NNFUSION_LOG(INFO) << "Avg Host duration:" << pctx->result.get_host_avg();
                NNFUSION_LOG(INFO) << "Avg Device duration:" << pctx->result.get_device_avg();
                auto res = prof.execute(inputs, element::f32);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(nnfusion::test::all_close_f(res[i], outputs[i]));
            }
            // Cuda
            if (true /*Check Cuda Runtime*/)
            {
                // Now we use the tool to profile and test the kernel;
                pctx->reset();
                NNFUSION_LOG(INFO) << "Test Cuda runtime of Pad operator:";
                nnfusion::profiler::Profiler prof(nnfusion::profiler::CudaDefaultRuntime::Runtime(),
                                                  pctx);
                prof.execute();
                NNFUSION_LOG(INFO) << "Avg Host duration:" << pctx->result.get_host_avg();
                NNFUSION_LOG(INFO) << "Avg Device duration:" << pctx->result.get_device_avg();

                auto res = prof.execute(inputs, element::f32);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(nnfusion::test::all_close_f(res[i], outputs[i]));
            }

            // Cpu Reference
            if (true /*Must support*/)
            {
                pctx->reset();
                NNFUSION_LOG(INFO) << "Test CPU Reference runtime of Pad operator:";
                nnfusion::profiler::Profiler ref_prof(
                    nnfusion::profiler::ReferenceRuntime::Runtime(), pctx);
                auto res = ref_prof.execute(inputs, element::f32);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(nnfusion::test::all_close_f(res[i], outputs[i]));
            }
        }
    }

    EXPECT_TRUE(has_valid_kernel);
}