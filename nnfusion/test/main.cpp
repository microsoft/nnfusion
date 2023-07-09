// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <iostream>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace std;

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

DECLARE_bool(fantares_mode);

int main(int argc, char** argv)
{
#ifdef NGRAPH_DISTRIBUTED
    ngraph::Distributed dist;
#endif
    google::SetUsageMessage(argv[0]);
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, false);

    if (FLAGS_fantares_mode)
    {
        for (auto pair : nnfusion::op::get_op_configs())
        {
            std::string op_name = pair.first;
            nnfusion::kernels::KernelRegistrar kernel_registrar_cuda(
                op_name,
                nnfusion::kernels::Name(op_name)
                    .Device(CUDA_GPU)
                    .TypeConstraint(element::f32)
                    .Tag("antares")
                    .Priority(9)
                    .KernelFactory([](shared_ptr<nnfusion::kernels::KernelContext> context)
                                       -> shared_ptr<nnfusion::kernels::KernelEmitter> {
                        return make_shared<nnfusion::kernels::cuda::AntaresCudaKernelEmitter>(
                            context);
                    })
                    .Build());
        }
    }
    // const char* exclude = "--gtest_filter=-benchmark.*";
    // vector<char*> argv_vector;
    // argv_vector.push_back(argv[0]);
    // argv_vector.push_back(const_cast<char*>(exclude));
    // for (int i = 1; i < argc; i++)
    // {
    //     argv_vector.push_back(argv[i]);
    // }
    // argc++;

    // ::testing::InitGoogleTest(&argc, argv_vector.data());
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();

    return rc;
}
