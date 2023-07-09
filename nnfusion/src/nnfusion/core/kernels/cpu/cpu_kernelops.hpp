// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            struct CpuOpMap;

            template <>
            struct CpuOpMap<nnfusion::op::Abs>
            {
                static constexpr const char* antares_op = "topi.abs";
                static constexpr const char* eigen_op = "eigen_abs";
                static constexpr const char* eigen_math_kernel = "in0.abs()";
                static constexpr const char* simd_op = "_mm256_abs_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), in0), in0)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Acos>
            {
                static constexpr const char* eigen_op = "eigen_acos";
                static constexpr const char* eigen_math_kernel = "in0.acos()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Asin>
            {
                static constexpr const char* eigen_op = "eigen_asin";
                static constexpr const char* eigen_math_kernel = "in0.asin()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Atan>
            {
                static constexpr const char* eigen_op = "eigen_atan";
                static constexpr const char* eigen_math_kernel = "in0.atan()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Ceiling>
            {
                static constexpr const char* antares_op = "topi.ceil";
                static constexpr const char* eigen_op = "eigen_ceil";
                static constexpr const char* eigen_math_kernel = "in0.ceil()";
                static constexpr const char* simd_op = "_mm256_ceil_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Cos>
            {
                static constexpr const char* antares_op = "topi.cos";
                static constexpr const char* eigen_op = "eigen_cos";
                static constexpr const char* eigen_math_kernel = "in0.cos()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Cosh>
            {
                static constexpr const char* eigen_op = "eigen_cosh";
                static constexpr const char* eigen_math_kernel = "in0.cosh()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Exp>
            {
                static constexpr const char* antares_op = "topi.exp";
                static constexpr const char* eigen_op = "eigen_exp";
                static constexpr const char* eigen_math_kernel = "in0.exp()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Floor>
            {
                static constexpr const char* antares_op = "topi.floor";
                static constexpr const char* eigen_op = "eigen_floor";
                static constexpr const char* eigen_math_kernel = "in0.floor()";
                static constexpr const char* simd_op = "_mm256_floor_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Log>
            {
                static constexpr const char* antares_op = "topi.log";
                static constexpr const char* eigen_op = "eigen_log";
                static constexpr const char* eigen_math_kernel = "in0.log()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Maximum>
            {
                static constexpr const char* antares_op = "topi.maximum";
                static constexpr const char* simd_op = "_mm256_max_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Minimum>
            {
                static constexpr const char* antares_op = "topi.minimum";
                static constexpr const char* simd_op = "_mm256_min_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Sin>
            {
                static constexpr const char* antares_op = "topi.sin";
                static constexpr const char* eigen_op = "eigen_sin";
                static constexpr const char* eigen_math_kernel = "in0.sin()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Sinh>
            {
                static constexpr const char* eigen_op = "eigen_sinh";
                static constexpr const char* eigen_math_kernel = "in0.sinh()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Sqrt>
            {
                static constexpr const char* antares_op = "topi.sqrt";
                static constexpr const char* eigen_op = "eigen_sqrt";
                static constexpr const char* eigen_math_kernel = "in0.sqrt()";
                static constexpr const char* simd_op = "_mm256_sqrt_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Rsqrt>
            {
                static constexpr const char* antares_op = "topi.rsqrt";
                static constexpr const char* eigen_op = "eigen_rsqrt";
                static constexpr const char* eigen_math_kernel = "in0.rsqrt()";
                static constexpr const char* simd_op = "_mm256_rsqrt_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Square>
            {
                static constexpr const char* antares_op = nullptr;
                static constexpr const char* eigen_op = "eigen_square";
                static constexpr const char* eigen_math_kernel = "in0.square()";
                static constexpr const char* simd_op = "_mm256_square_ps";
                static constexpr const char* simd_math_kernel = "_mm256_mul_ps(in0, in0)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Tan>
            {
                static constexpr const char* eigen_op = "eigen_tan";
                static constexpr const char* eigen_math_kernel = "in0.tan()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Tanh>
            {
                static constexpr const char* antares_op = "topi.tanh";
                static constexpr const char* eigen_op = "eigen_tanh";
                static constexpr const char* eigen_math_kernel = "in0.tanh()";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Power>
            {
                static constexpr const char* antares_op = "topi.power";
                static constexpr const char* eigen_op = "eigen_pow";
                static constexpr const char* eigen_math_kernel = "in0.pow(in1)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Subtract>
            {
                static constexpr const char* antares_op = "topi.subtract";
                static constexpr const char* eigen_op = "eigen_subtract";
                static constexpr const char* eigen_math_kernel = "in0 - in1";
                static constexpr const char* simd_op = "_mm256_sub_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Divide>
            {
                static constexpr const char* antares_op = "topi.divide";
                static constexpr const char* eigen_op = "eigen_divide";
                static constexpr const char* eigen_math_kernel = "in0 / in1";
                static constexpr const char* simd_op = "_mm256_div_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::DivNoNan>
            {
                static constexpr const char* antares_op = "divnonan";
                static constexpr const char* math_kernel = "x1 != 0 ? fdividef(x0, x1) : 0";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sign>
            {
                static constexpr const char* antares_op = "topi.sign";
                static constexpr const char* eigen_op = "eigen_sign";
                static constexpr const char* eigen_math_kernel = "in0.sign()";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Convert>
            {
                static constexpr const char* antares_op = "convert";
                static constexpr const char* math_kernel = "x0";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Equal>
            {
                static constexpr const char* antares_op = "topi.equal";
                static constexpr const char* simd_op = "_mm256_cmpeq_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_EQ_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::NotEqual>
            {
                static constexpr const char* antares_op = "topi.not_equal";
                static constexpr const char* simd_op = "_mm256_cmpneq_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_NEQ_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Greater>
            {
                static constexpr const char* antares_op = "topi.greater";
                static constexpr const char* simd_op = "_mm256_cmpgt_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_GT_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::GreaterEq>
            {
                static constexpr const char* antares_op = "topi.greater_equal";
                static constexpr const char* simd_op = "_mm256_cmpge_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_GE_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Less>
            {
                static constexpr const char* antares_op = "topi.less";
                static constexpr const char* simd_op = "_mm256_cmplt_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_LT_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::LessEq>
            {
                static constexpr const char* antares_op = "topi.less_equal";
                static constexpr const char* simd_op = "_mm256_cmple_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_cmp_ps(in0, in1, _CMP_LE_OS)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Relu>
            {
                static constexpr const char* antares_op = "topi.nn.relu";
                static constexpr const char* simd_op = "_mm256_cmple_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_max_ps(_mm256_setzero_ps(), in0)";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Not>
            {
                static constexpr const char* antares_op = "topi.logical_not";
                static constexpr const char* simd_op = "_mm256_not_ps";
                static constexpr const char* simd_math_kernel = "~x0";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Negative>
            {
                static constexpr const char* antares_op = "topi.negative";
                static constexpr const char* simd_op = "_mm256_neg_ps";
                static constexpr const char* simd_math_kernel =
                    "_mm256_xor_ps(in0, _mm256_set1_ps(-0.0f))";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::ReluBackprop>
            {
                static constexpr const char* antares_op = "relu_backprop";
                static constexpr const char* math_kernel = "x1 * int(x0 > 0)";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::And>
            {
                static constexpr const char* antares_op = "topi.logical_and";
                static constexpr const char* simd_op = "_mm256_and_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Or>
            {
                static constexpr const char* antares_op = "topi.logical_or";
                static constexpr const char* simd_op = "_mm256_or_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Add>
            {
                static constexpr const char* antares_op = "topi.add";
                static constexpr const char* eigen_op = "eigen_add";
                static constexpr const char* eigen_math_kernel = "in0 + in1";
                static constexpr const char* simd_op = "_mm256_add_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Multiply>
            {
                static constexpr const char* antares_op = "topi.multiply";
                static constexpr const char* eigen_op = "eigen_multiply";
                static constexpr const char* eigen_math_kernel = "in0 * in1";
                static constexpr const char* simd_op = "_mm256_mul_ps";
                static constexpr const char* simd_math_kernel = nullptr;
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Nop>
            {
                static constexpr const char* antares_op = "";
                static constexpr const char* math_kernel = "";
                static constexpr const char* atomic = "";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sigmoid>
            {
                static constexpr const char* antares_op = "topi.sigmoid";
                static constexpr const char* eigen_op = "eigen_sigmoid";
                static constexpr const char* eigen_math_kernel = "1/(1+(-in0).exp())";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::SigmoidBackprop>
            {
                static constexpr const char* antares_op = "sigmoid_backprop";
                static constexpr const char* math_kernel = "x1 / (2 + expf(-x0) + expf(x0))";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sum>
            {
                static constexpr const char* antares_op = "topi.sum";
                static constexpr const char* eigen_op = "sum";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Product>
            {
                static constexpr const char* antares_op = "topi.prod";
                static constexpr const char* eigen_op = "prod";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Max>
            {
                static constexpr const char* antares_op = "topi.max";
                static constexpr const char* eigen_op = "maxCoeff";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Min>
            {
                static constexpr const char* antares_op = "topi.min";
                static constexpr const char* eigen_op = "minCoeff";
            };
        }
    }
}
