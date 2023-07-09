// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "nnfusion/test_util/all_close.hpp"
#include "nnfusion/test_util/inventory.hpp"
#include "nnfusion/test_util/ndarray.hpp"

#define EXPECT_POINTER_TYPE(pointer, type, new_pointer)                                            \
    auto new_pointer = static_pointer_cast<type>(pointer);                                         \
    EXPECT_TRUE(new_pointer != nullptr);

using namespace std;
using namespace nnfusion;
using namespace nnfusion::profiler;

template <typename t>
void print_vector(const vector<t>& v, string v_name)
{
    cout << v_name << " = {";
    for (auto& e : v)
        cout << e << ", ";
    cout << "};\n";
}

template <typename t>
void print_set(const set<t>& v, string v_name)
{
    cout << v_name << " = {";
    for (auto& e : v)
        cout << e << ", ";
    cout << "};\n";
}

template <typename t, typename p>
bool compare_vector(const vector<t>& a, const vector<p> b)
{
    if (a.size() != b.size())
        return false;
    for (int i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            return false;
    return true;
}

namespace nnfusion
{
    namespace test
    {
        bool check_kernel(shared_ptr<GNode> gnode,
                          NNFusion_DeviceType dev_t,
                          const vector<float>& IN,
                          const vector<float>& OUT);

        template <typename T = float>
        bool check_kernel(shared_ptr<GNode> gnode,
                          NNFusion_DeviceType dev_t,
                          const vector<T>& IN,
                          const vector<T>& OUT)
        {
            auto rt = get_default_runtime(dev_t);
            if (rt == nullptr)
            {
                return false;
            }
            std::vector<shared_ptr<const KernelRegistration>> available_kernels =
                KernelRegistry::Global()->FindKernelRegistrations(
                    gnode->get_op_type(), dev_t, element::f32);
            shared_ptr<KernelContext> ctx(new KernelContext(gnode));
            bool kernel_found = false;
            for (auto& kernel_reg : available_kernels)
            {
                // TODO: Eigen kernel uses tensor_layout in codegen while tensor_layout doesn't be set in profiler
                if (kernel_reg->m_tag == "eigen")
                    continue;
                auto kernel = kernel_reg->m_factory(ctx);
                if (kernel->get_or_emit_source())
                {
                    kernel_found = true;
                    auto pctx = make_shared<ProfilingContext>(kernel);
                    pctx->runtime_times = 1;
                    pctx->warmup_times = 0;
                    Profiler prof(rt, pctx);

                    // The execute() will return a vector of vector,
                    // we only compare the first one with our ground
                    // truth
                    auto res = prof.unsafe_execute<T>((void*)IN.data());
                    if (res.empty())
                    {
                        NNFUSION_LOG(INFO) << "Kernel empty result.";
                        return false;
                    }
                    auto& res_first = res[0];

                    if (res_first.size() != OUT.size())
                    {
                        NNFUSION_LOG(INFO) << "Kernel result size error:" << res_first.size()
                                           << " vs. " << OUT.size();
                        return false;
                    }

                    if (!all_close<T>(res_first, OUT))
                        return false;

                    NNFUSION_LOG(INFO) << "Kernel with tag '" << kernel_reg->m_tag
                                       << "' pass unit-test.";
                }
                else
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "Kernel with tag '" << kernel_reg->m_tag
                                                   << "' is not available.";
                }
            }
            if (!kernel_found)
            {
                NNFUSION_LOG(ERROR) << "There is no available kernel found!";
            }
            return kernel_found;
        }
    }
}
