// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::inventory;
using namespace nnfusion::profiler;

DECLARE_bool(fantares_mode);
namespace nnfusion
{
    namespace test
    {
        ///\todo Maybe a better/general way

        template <typename T, typename val_t = float>
        bool check_kernels(NNFusion_DeviceType dev_t, element::Type data_t)
        {
            for (int case_id = 0;; case_id++)
            {
                auto gnode = create_object<T, val_t>(case_id);
                if (gnode == nullptr)
                    break;
                NNFUSION_LOG(INFO) << "TestOp: " << gnode->get_op_type() << ", CaseId: " << case_id;
                auto input = generate_input<T, val_t>(case_id);
                auto output = generate_output<T, val_t>(case_id);

                bool result = nnfusion::test::check_kernel<val_t>(gnode, dev_t, input, output);
                if (!result)
                {
                    NNFUSION_LOG(ERROR) << "Kernel test failed for test case: " << case_id;
                    return false;
                }
            }
            return true;
        }
    }
}

///param: node, device_type, data_type ... etc
TEST(nnfusion_core_kernels, batch_kernel_tests_abs)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Abs>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Abs>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_add)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Add>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Add>(CUDA_GPU, element::f32));
}

/* TODO: arg type is bool, enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_and)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::And>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::And>(CUDA_GPU, element::f32));
}
*/

/* TODO: arg index type is i32/i64, enable if more data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_arg_max)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ArgMax>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ArgMax>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_arg_min)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ArgMin>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ArgMin>(CUDA_GPU, element::f32));
}
*/

TEST(nnfusion_core_kernels, batch_kernel_tests_broadcast)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Broadcast>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Broadcast>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_ceiling)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Ceiling>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Ceiling>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_concat)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Concat>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Concat>(CUDA_GPU, element::f32));
}

/* TODO: enable if more data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_convert)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Convert>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Convert>(CUDA_GPU, element::f32));
}
*/

TEST(nnfusion_core_kernels, batch_kernel_tests_divide)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Divide>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Divide>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_dot)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Dot>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Dot>(CUDA_GPU, element::f32));
}

/* TODO: return type is bool, enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_equal)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Equal>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Equal>(CUDA_GPU, element::f32));
}
*/
TEST(nnfusion_core_kernels, batch_kernel_tests_floor)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Floor>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Floor>(CUDA_GPU, element::f32));
}

/* TODO: return type is bool, enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_greater)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Greater>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Greater>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_greater_eq)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::GreaterEq>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::GreaterEq>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_less)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Less>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Less>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_less_eq)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::LessEq>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::LessEq>(CUDA_GPU, element::f32));
}
*/

TEST(nnfusion_core_kernels, batch_kernel_tests_max)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Max>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Max>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_max_pool)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::MaxPool>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::MaxPool>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_maximum)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Maximum>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Maximum>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_min)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Min>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Min>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_minimum)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Minimum>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Minimum>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_multiply)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Multiply>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Multiply>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_negative)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Negative>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Negative>(CUDA_GPU, element::f32));
}

/* TODO: enable if more data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_not)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Not>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Not>(CUDA_GPU, element::f32));
}
*/
/* TODO: return type is bool, enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_not_equal)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::NotEqual>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::NotEqual>(CUDA_GPU, element::f32));
}
*/
/* TODO: enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_or)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Or>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Or>(CUDA_GPU, element::f32));
}
*/
TEST(nnfusion_core_kernels, batch_kernel_tests_pad)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Pad>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Pad>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_product)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Product>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Product>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_relu)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Relu>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Relu>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_relu_backprop)
{
    // TODO: there is no cpu kernel implemented
    // EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ReluBackprop>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ReluBackprop>(CUDA_GPU, element::f32));
}

/* TODO: there is no replace slice kernel implemented
TEST(nnfusion_core_kernels, batch_kernel_tests_replace_slice)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ReplaceSlice>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::ReplaceSlice>(CUDA_GPU, element::f32));
}
*/
TEST(nnfusion_core_kernels, batch_kernel_tests_reshape)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Reshape>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Reshape>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_reverse)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Reverse>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Reverse>(CUDA_GPU, element::f32));
}

/* TODO: enable if bool data type is supported, the test case data type should also be modified 
TEST(nnfusion_core_kernels, batch_kernel_tests_select)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Select>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Select>(CUDA_GPU, element::f32));
}
*/
TEST(nnfusion_core_kernels, batch_kernel_tests_sign)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sign>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sign>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_slice)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Slice>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Slice>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_sqrt)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sqrt>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sqrt>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_subtract)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Subtract>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Subtract>(CUDA_GPU, element::f32));
}

TEST(nnfusion_core_kernels, batch_kernel_tests_sum)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sum>(GENERIC_CPU, element::f32));
    EXPECT_TRUE(nnfusion::test::check_kernels<nnfusion::op::Sum>(CUDA_GPU, element::f32));
}