// Microsoft (c) 2019, MSRA/NNFUSION Team
///\brief Batch tests for our kernels.
///
///\author wenxh, ziming

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/core/operators/op_define/abs.hpp"
#include "nnfusion/core/operators/op_define/add.hpp"
#include "nnfusion/core/operators/op_define/and.hpp"
#include "nnfusion/core/operators/op_define/argmax.hpp"
#include "nnfusion/core/operators/op_define/argmin.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/ceiling.hpp"
#include "nnfusion/core/operators/op_define/concat.hpp"
#include "nnfusion/core/operators/op_define/convert.hpp"
#include "nnfusion/core/operators/op_define/divide.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/core/operators/op_define/equal.hpp"
#include "nnfusion/core/operators/op_define/floor.hpp"
#include "nnfusion/core/operators/op_define/greater.hpp"
#include "nnfusion/core/operators/op_define/greater_eq.hpp"
#include "nnfusion/core/operators/op_define/less.hpp"
#include "nnfusion/core/operators/op_define/less_eq.hpp"
#include "nnfusion/core/operators/op_define/max.hpp"
#include "nnfusion/core/operators/op_define/max_pool.hpp"
#include "nnfusion/core/operators/op_define/maximum.hpp"
#include "nnfusion/core/operators/op_define/min.hpp"
#include "nnfusion/core/operators/op_define/minimum.hpp"
#include "nnfusion/core/operators/op_define/multiply.hpp"
#include "nnfusion/core/operators/op_define/negative.hpp"
#include "nnfusion/core/operators/op_define/not.hpp"
#include "nnfusion/core/operators/op_define/not_equal.hpp"
#include "nnfusion/core/operators/op_define/one_hot.hpp"
#include "nnfusion/core/operators/op_define/or.hpp"
#include "nnfusion/core/operators/op_define/pad.hpp"
#include "nnfusion/core/operators/op_define/product.hpp"
#include "nnfusion/core/operators/op_define/relu.hpp"
#include "nnfusion/core/operators/op_define/relu.hpp"
#include "nnfusion/core/operators/op_define/replace_slice.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/op_define/reverse.hpp"
#include "nnfusion/core/operators/op_define/select.hpp"
#include "nnfusion/core/operators/op_define/sign.hpp"
#include "nnfusion/core/operators/op_define/slice.hpp"
#include "nnfusion/core/operators/op_define/sqrt.hpp"
#include "nnfusion/core/operators/op_define/subtract.hpp"
#include "nnfusion/core/operators/op_define/sum.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::inventory;
using namespace nnfusion::profiler;

namespace nnfusion
{
    namespace test
    {
        template <>
        bool all_close<float>(const std::vector<float>& a, const std::vector<float>& b)
        {
            return all_close_f(a, b);
        }

        template <>
        bool all_close<int>(const std::vector<int>& a, const std::vector<int>& b)
        {
            if (a.size() == b.size())
            {
                for (size_t i = 0; i < a.size(); i++)
                    if (a[i] != b[i])
                        return false;
                return true;
            }
            return false;
        }

        ///\todo Maybe a better/general way

        template <typename T, typename val_t = float>
        bool check_kernels(DeviceType dev_t, DataType data_t)
        {
            auto rt = get_default_runtime(dev_t);
            if (rt == nullptr)
                return false;

            for (int case_id = 0;; case_id++)
            {
                auto gnode = create_object<T, val_t>(case_id);
                if (gnode == nullptr)
                    break;
                LOG(INFO) << "TestOp: " << gnode->get_op_type() << ", CaseId: " << case_id;
                auto input = generate_input<T, val_t>(case_id);
                auto output = generate_output<T, val_t>(case_id);
                shared_ptr<KernelContext> ctx(new KernelContext(gnode));
                auto available_kernels = KernelRegistry::Global()->FindKernelRegistrations(
                    gnode->get_op_type(), dev_t, data_t);

                for (auto& kernel_reg : available_kernels)
                {
                    auto kernel = kernel_reg->m_factory(ctx);
                    if (kernel->get_or_emit_source())
                    {
                        auto pctx = make_shared<ProfilingContext>(kernel);
                        Profiler prof(rt, pctx);

                        // The execute() will return a vector of vector,
                        // we only compare the first one with our ground
                        // truth
                        auto res = prof.unsafe_execute<val_t>((void*)input.data());
                        if (res.empty())
                            return false;
                        auto& res_first = res[0];

                        if (res_first.size() != output.size())
                            return false;

                        if (!all_close_f(res_first, output))
                            return false;
                    }
                    else
                    {
                        LOG(WARNING) << "Kernel is not available";
                    }
                }
            }
            return true;
        }
    }
}

///param: node, device_type, data_type ... etc
TEST(nnfusion_core_kernels, batch_kernel_tests_abs)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Abs>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Abs>(CUDA_GPU, DT_FLOAT));
}
/*
TEST(nnfusion_core_kernels, batch_kernel_tests_add)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Add>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Add>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_and)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::And>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::And>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_arg_max)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ArgMax>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ArgMax>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_arg_min)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ArgMin>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ArgMin>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_broadcast)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Broadcast>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Broadcast>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_ceiling)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Ceiling>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Ceiling>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_concat)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Concat>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Concat>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_convert)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Convert>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Convert>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_divide)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Divide>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Divide>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_dot)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Dot>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Dot>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_equal)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Equal>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Equal>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_floor)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Floor>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Floor>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_greater)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Greater>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Greater>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_greater_eq)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::GreaterEq>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::GreaterEq>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_less)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Less>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Less>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_less_eq)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::LessEq>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::LessEq>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_max)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Max>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Max>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_max_pool)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::MaxPool>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::MaxPool>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_maximum)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Maximum>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Maximum>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_min)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Min>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Min>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_minimum)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Minimum>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Minimum>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_multiply)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Multiply>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Multiply>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_negative)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Negative>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Negative>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_not)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Not>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Not>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_not_equal)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::NotEqual>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::NotEqual>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_one_hot)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::OneHot>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::OneHot>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_or)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Or>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Or>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_pad)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Pad>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Pad>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_product)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Product>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Product>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_relu)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Relu>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Relu>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_relu_backprop)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ReluBackprop>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::ReluBackprop>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_replace_slice)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::ReplaceSlice>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::ReplaceSlice>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_reshape)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Reshape>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Reshape>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_reverse)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Reverse>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Reverse>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_select)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Select>(GENERIC_CPU, DT_FLOAT));
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Select>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_sign)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Sign>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Sign>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_slice)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Slice>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Slice>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_sqrt)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Sqrt>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Sqrt>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_subtract)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Subtract>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Subtract>(CUDA_GPU, DT_FLOAT));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_sum)
{
    // EXPECT_TRUE(nnfusion::test::check_kernels<op::Sum>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Sum>(CUDA_GPU, DT_FLOAT));
}
*/