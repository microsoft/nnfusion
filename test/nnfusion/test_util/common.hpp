// Microsoft (c) 2019, Wenxiang
#pragma once

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "all_close.hpp"
#include "inventory.hpp"
#include "ndarray.hpp"

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
        template <typename T>
        extern bool all_close(const std::vector<T>& a, const std::vector<T>& b);

        template <>
        bool all_close<float>(const std::vector<float>& a, const std::vector<float>& b);

        template <>
        bool all_close<int>(const std::vector<int>& a, const std::vector<int>& b);

        bool check_kernel(shared_ptr<GNode> gnode,
                          DeviceType dev_t,
                          const vector<float>& IN,
                          const vector<float>& OUT);

        template <typename T = float>
        bool check_kernel(shared_ptr<GNode> gnode,
                          DeviceType dev_t,
                          const vector<T>& IN,
                          const vector<T>& OUT)
        {
            auto rt = get_default_runtime(dev_t);
            std::vector<shared_ptr<const KernelRegistration>> available_kernels =
                KernelRegistry::Global()->FindKernelRegistrations(
                    gnode->get_op_type(), dev_t, DT_FLOAT);
            shared_ptr<KernelContext> ctx(new KernelContext(gnode));
            bool kernel_found = false;
            for (auto& kernel_reg : available_kernels)
            {
                auto kernel = kernel_reg->m_factory(ctx);
                if (kernel->get_or_emit_source())
                {
                    kernel_found = true;
                    auto pctx = make_shared<ProfilingContext>(kernel);
                    Profiler prof(rt, pctx);

                    // The execute() will return a vector of vector,
                    // we only compare the first one with our ground
                    // truth
                    auto res = prof.unsafe_execute<T>((void*)IN.data());
                    if (res.empty())
                    {
                        LOG(INFO) << "Kernel empty result.";
                        return false;
                    }
                    auto& res_first = res[0];

                    if (res_first.size() != OUT.size())
                    {
                        LOG(INFO) << "Kernel result size error:" << res_first.size() << " vs. "
                                  << OUT.size();
                        return false;
                    }

                    if (!all_close(res_first, OUT))
                        return false;

                    LOG(INFO) << "Kernel pass unit-test.";
                }
                else
                {
                    LOG(WARNING) << "Kernel is not available.";
                }
            }
            if (!kernel_found)
                LOG(WARNING) << "No available found!";
            return kernel_found;
        }
    }
}