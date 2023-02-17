//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "op/adam_optimizer.hpp"
#include "op/attention.hpp"
#include "op/batch_norm.hpp"
#include "op/bias_gelu.hpp"
#include "op/binaryop.hpp"
#include "op/cast.hpp"
#include "op/cast_like.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/const_of_shape.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/conv_trans.hpp"
#include "op/cum_sum.hpp"
#include "op/depth_to_space.hpp"
#include "op/div_grad.hpp"
#include "op/dropout.hpp"
#include "op/einsum.hpp"
#include "op/embed_layer_norm.hpp"
#include "op/erf_grad.hpp"
#include "op/expand.hpp"
#include "op/flatten.hpp"
#include "op/gather.hpp"
#include "op/gemm.hpp"
#include "op/gru.hpp"
#include "op/identity.hpp"
#include "op/index_reduce.hpp"
#include "op/instance_norm.hpp"
#include "op/layer_norm.hpp"
#include "op/leaky_relu.hpp"
#include "op/log_softmax.hpp"
#include "op/lstm.hpp"
#include "op/matmul.hpp"
#include "op/mean.hpp"
#include "op/memory_copy.hpp"
#include "op/multi_elementwise.hpp"
#include "op/non_zero.hpp"
#include "op/one_hot.hpp"
#include "op/pad.hpp"
#include "op/pool.hpp"
#include "op/range.hpp"
#include "op/reciprocal.hpp"
#include "op/reduce.hpp"
#include "op/reshape.hpp"
#include "op/resize.hpp"
#include "op/roll.hpp"
#include "op/scatternd.hpp"
#include "op/shape.hpp"
#include "op/size.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/split.hpp"
#include "op/squeeze.hpp"
#include "op/sum.hpp"
#include "op/tanh_grad.hpp"
#include "op/tile.hpp"
#include "op/transpose.hpp"
#include "op/unaryop.hpp"
#include "op/unsqueeze.hpp"
#include "op/where.hpp"
#include "ops_bridge.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace detail
            {
                const ConvertFunc& find(const std::string& name,
                                        std::int64_t version,
                                        const std::string& domain,
                                        const std::map<std::int64_t, ConvertFunc>& map)
                {
                    const static ConvertFunc EMPTY_FUNC = nullptr;
                    int64_t avail_version = version;
                    while (avail_version > 0)
                    {
                        const auto it = map.find(avail_version--);
                        if (it != std::end(map))
                        {
                            return it->second;
                        }
                    }
                    return EMPTY_FUNC;
                }
            } // namespace detail

            void OperatorsBridge::_register_operator(const std::string& name,
                                                     std::int64_t version,
                                                     const std::string& domain,
                                                     ConvertFunc fn)
            {
                m_map[domain][name].emplace(version, std::move(fn));
            }

            ConvertFuncMap OperatorsBridge::_get_convert_func_map(std::int64_t version,
                                                                  const std::string& domain)
            {
                ConvertFuncMap result;
                auto dm = m_map.find(domain);
                if (dm != std::end(m_map))
                {
                    for (const auto& op : dm->second)
                    {
                        const auto& convert_func =
                            detail::find(op.first, version, domain, op.second);
                        if (convert_func)
                        {
                            result.emplace(op.first, convert_func);
                        }
                    }
                }
                return result;
            }
#define PACK(...) __VA_ARGS__
#define REGISTER_DOMAIN_OPERATOR(domain_, name_, ver_, fn_)                                        \
    m_map[domain_][name_].emplace(                                                                 \
        ver_,                                                                                      \
        std::bind(                                                                                 \
            set_##ver_::fn_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3))

#define REGISTER_OPERATOR(name_, ver_, fn_) REGISTER_DOMAIN_OPERATOR("", name_, ver_, PACK(fn_))

#define REGISTER_EMPTY_DOMAIN(domain_) m_map[domain_]

            OperatorsBridge::OperatorsBridge()
            {
                REGISTER_OPERATOR("Abs", 1, TranslateUnaryOp<op::Abs>);
                REGISTER_OPERATOR("Acos", 1, TranslateUnaryOp<op::Acos>);
                REGISTER_OPERATOR("AveragePool", 1, TranslateAveragePoolOp);
                REGISTER_OPERATOR("AveragePool", 7, TranslateAveragePoolOp);
                REGISTER_OPERATOR("AveragePool", 10, TranslateAveragePoolOp);
                REGISTER_OPERATOR("AveragePool", 11, TranslateAveragePoolOp);
                REGISTER_OPERATOR("AdamOptimizer", 1, TranslateAdamOptimizerOp);
                REGISTER_OPERATOR("Add", 1, TranslateLegacyBinaryOp<op::Add>);
                REGISTER_OPERATOR("Add", 7, TranslateBinaryOp<op::Add>);
                REGISTER_OPERATOR("And", 1, TranslateBinaryOp<op::And>);
                REGISTER_OPERATOR("ArgMin", 1, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMin", 11, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMin", 12, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMin", 13, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMin", 14, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMax", 1, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("ArgMax", 11, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("ArgMax", 12, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("ArgMax", 13, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("ArgMax", 14, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("Asin", 1, TranslateUnaryOp<op::Asin>);
                REGISTER_OPERATOR("Atan", 1, TranslateUnaryOp<op::Atan>);
                REGISTER_DOMAIN_OPERATOR("com.microsoft", "Attention", 1, TranslateAttentionOp);
                REGISTER_OPERATOR("BatchNormalization", 1, TranslateBatchNormOp);
                REGISTER_OPERATOR("BatchNormalization", 6, TranslateBatchNormOp);
                REGISTER_OPERATOR("BatchNormalization", 7, TranslateBatchNormOp);
                REGISTER_OPERATOR("BatchNormalization", 9, TranslateBatchNormOp);
                REGISTER_OPERATOR("BatchNormalization", 14, TranslateBatchNormOp);
                REGISTER_OPERATOR("BatchNormalization", 15, TranslateBatchNormOp);
                REGISTER_DOMAIN_OPERATOR("com.microsoft", "BiasGelu", 1, TranslateBiasGeluOp);
                REGISTER_OPERATOR("Cast", 1, TranslateCastOp);
                REGISTER_OPERATOR("Cast", 6, TranslateCastOp);
                REGISTER_OPERATOR("Cast", 9, TranslateCastOp);
                REGISTER_OPERATOR("Cast", 13, TranslateCastOp);
                REGISTER_OPERATOR("CastLike", 15, TranslateCastLikeOp);
                REGISTER_OPERATOR("Ceil", 1, TranslateUnaryOp<op::Ceiling>);
                REGISTER_OPERATOR("Ceil", 6, TranslateUnaryOp<op::Ceiling>);
                REGISTER_OPERATOR("Ceil", 13, TranslateUnaryOp<op::Ceiling>);
                REGISTER_OPERATOR("Clip", 1, TranslateClipOp);
                REGISTER_OPERATOR("Clip", 6, TranslateClipOp);
                REGISTER_OPERATOR("Clip", 11, TranslateClipOp);
                REGISTER_OPERATOR("Clip", 12, TranslateClipOp);
                REGISTER_OPERATOR("Clip", 13, TranslateClipOp);
                REGISTER_OPERATOR("Concat", 1, TranslateConcatOp);
                REGISTER_OPERATOR("Concat", 4, TranslateConcatOp);
                REGISTER_OPERATOR("Concat", 11, TranslateConcatOp);
                REGISTER_OPERATOR("Concat", 13, TranslateConcatOp);
                REGISTER_OPERATOR("Constant", 1, TranslateConstantOp);
                REGISTER_OPERATOR("Constant", 9, TranslateConstantOp);
                REGISTER_OPERATOR("ConstantOfShape", 9, TranslateConstantOfShapeOp);
                REGISTER_OPERATOR("Conv", 1, TranslateConvOp);
                REGISTER_OPERATOR("Conv", 11, TranslateConvOp);
                REGISTER_OPERATOR("Cos", 7, TranslateUnaryOp<op::Cos>);
                REGISTER_OPERATOR("CumSum", 11, TranslateCumSumOp);
                REGISTER_OPERATOR("CumSum", 14, TranslateCumSumOp);
                REGISTER_OPERATOR("Div", 1, TranslateLegacyBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 6, TranslateBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 7, TranslateBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 13, TranslateBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 14, TranslateBinaryOp<op::Divide>);
                REGISTER_OPERATOR("DivGrad", 1, TranslateDivGradOp);
                REGISTER_OPERATOR("Dropout", 1, TranslateDropoutOp);
                REGISTER_OPERATOR("Dropout", 6, TranslateDropoutOp);
                REGISTER_OPERATOR("Dropout", 7, TranslateDropoutOp);
                REGISTER_OPERATOR("Dropout", 10, TranslateDropoutOp);
                REGISTER_OPERATOR("Dropout", 12, TranslateDropoutOp);
                REGISTER_OPERATOR("Dropout", 13, TranslateDropoutOp);
                //REGISTER_OPERATOR("Elu", 1, elu);
                REGISTER_OPERATOR("Einsum", 12, TranslateEinsumOp);
                REGISTER_DOMAIN_OPERATOR(
                    "com.microsoft", "EmbedLayerNormalization", 1, TranslateEmbedLayerNormOp);
                REGISTER_OPERATOR("Equal", 1, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Equal", 7, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Equal", 11, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Equal", 13, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Erf", 9, TranslateUnaryOp<op::Erf>);
                REGISTER_OPERATOR("Erf", 13, TranslateUnaryOp<op::Erf>);
                REGISTER_OPERATOR("ErfGrad", 1, TranslateErfGradOp);
                REGISTER_OPERATOR("Exp", 1, TranslateUnaryOp<op::Exp>);
                REGISTER_OPERATOR("Exp", 6, TranslateUnaryOp<op::Exp>);
                REGISTER_OPERATOR("Exp", 13, TranslateUnaryOp<op::Exp>);
                REGISTER_OPERATOR("Expand", 8, TranslateExpandOp);
                REGISTER_OPERATOR("Expand", 13, TranslateExpandOp);
                REGISTER_OPERATOR("Flatten", 1, TranslateFlattenOp);
                REGISTER_OPERATOR("Flatten", 9, TranslateFlattenOp);
                REGISTER_OPERATOR("Flatten", 11, TranslateFlattenOp);
                REGISTER_OPERATOR("Flatten", 13, TranslateFlattenOp);
                REGISTER_OPERATOR("Floor", 1, TranslateUnaryOp<op::Floor>);
                REGISTER_OPERATOR("Floor", 6, TranslateUnaryOp<op::Floor>);
                REGISTER_OPERATOR("Floor", 13, TranslateUnaryOp<op::Floor>);
                REGISTER_OPERATOR("Gather", 1, TranslateGatherOp);
                REGISTER_OPERATOR("Gather", 11, TranslateGatherOp);
                REGISTER_OPERATOR("Gather", 13, TranslateGatherOp);
                REGISTER_OPERATOR("GatherGrad", 11, TranslateGatherGradOp);
                REGISTER_OPERATOR("GatherND", 11, TranslateGatherNDOp);
                REGISTER_OPERATOR("GatherND", 12, TranslateGatherNDOp);
                REGISTER_OPERATOR("GatherND", 13, TranslateGatherNDOp);
                REGISTER_OPERATOR("GatherNDGrad", 11, TranslateGatherNDGradOp);
                REGISTER_OPERATOR("Gelu", 1, TranslateUnaryOp<op::Gelu>);
                REGISTER_OPERATOR("GlobalAveragePool",
                                  1,
                                  PACK(TranslateGlobalPoolOp<op::NoOp, op::Sum, op::Divide>));
                REGISTER_OPERATOR(
                    "GlobalMaxPool", 1, PACK(TranslateGlobalPoolOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR("Greater", 1, TranslateLegacyBinaryOp<op::Greater>);
                REGISTER_OPERATOR("Greater", 7, TranslateBinaryOp<op::Greater>);
                REGISTER_OPERATOR("Greater", 9, TranslateBinaryOp<op::Greater>);
                REGISTER_OPERATOR("Greater", 13, TranslateBinaryOp<op::Greater>);
                REGISTER_OPERATOR("GreaterOrEqual", 12, TranslateBinaryOp<op::GreaterEq>);
                REGISTER_OPERATOR("GreaterOrEqual", 16, TranslateBinaryOp<op::GreaterEq>);
                REGISTER_OPERATOR("Gemm", 7, TranslateGemmOp);
                REGISTER_OPERATOR("Gemm", 9, TranslateGemmOp);
                REGISTER_OPERATOR("Gemm", 11, TranslateGemmOp);
                REGISTER_OPERATOR("Gemm", 13, TranslateGemmOp);
                REGISTER_OPERATOR("GRU", 7, TranslateGRUOp);
                REGISTER_OPERATOR("GRU", 14, TranslateGRUOp);
                //REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
                REGISTER_OPERATOR("Identity", 1, TranslateIdentityOp);
                REGISTER_OPERATOR("Identity", 13, TranslateIdentityOp);
                REGISTER_OPERATOR("Identity", 14, TranslateIdentityOp);
                REGISTER_OPERATOR("Identity", 16, TranslateIdentityOp);
                REGISTER_OPERATOR("InstanceNormalization", 1, TranslateInstanceNormalizationOp);
                REGISTER_OPERATOR("InstanceNormalization", 6, TranslateInstanceNormalizationOp);
                REGISTER_OPERATOR("LayerNormalization", 17, TranslateLayerNormalizationOp);
                REGISTER_OPERATOR("LayerNormalizationGrad", 1, TranslateLayerNormalizationGradOp);
                REGISTER_OPERATOR("LeakyRelu", 1, TranslateLeakyReluOp);
                REGISTER_OPERATOR("LeakyRelu", 6, TranslateLeakyReluOp);
                REGISTER_OPERATOR("LeakyRelu", 16, TranslateLeakyReluOp);
                REGISTER_OPERATOR("Less", 1, TranslateLegacyBinaryOp<op::Less>);
                REGISTER_OPERATOR("Less", 7, TranslateBinaryOp<op::Less>);
                REGISTER_OPERATOR("Less", 9, TranslateBinaryOp<op::Less>);
                REGISTER_OPERATOR("Less", 13, TranslateBinaryOp<op::Less>);
                REGISTER_OPERATOR("LessOrEqual", 12, TranslateBinaryOp<op::LessEq>);
                REGISTER_OPERATOR("LessOrEqual", 16, TranslateBinaryOp<op::LessEq>);
                REGISTER_OPERATOR("Log", 1, TranslateUnaryOp<op::Log>);
                REGISTER_OPERATOR("Log", 6, TranslateUnaryOp<op::Log>);
                REGISTER_OPERATOR("Log", 13, TranslateUnaryOp<op::Log>);
                REGISTER_OPERATOR("LogSoftmax", 1, TranslateLogSoftmaxOp);
                REGISTER_OPERATOR("LogSoftmax", 11, TranslateLogSoftmaxOp);
                REGISTER_OPERATOR("LogSoftmax", 13, TranslateLogSoftmaxOp);
                //REGISTER_OPERATOR("LRN", 1, lrn);
                REGISTER_OPERATOR("LSTM", 1, TranslateLstmOp);
                REGISTER_OPERATOR("LSTM", 7, TranslateLstmOp);
                REGISTER_OPERATOR("LSTM", 14, TranslateLstmOp);
                REGISTER_OPERATOR("MatMul", 1, TranslateMatmulOp);
                REGISTER_OPERATOR("MatMul", 9, TranslateMatmulOp);
                REGISTER_OPERATOR("MatMul", 13, TranslateMatmulOp);
                REGISTER_OPERATOR("MaxPool", 1, TranslateMaxPoolOp);
                REGISTER_OPERATOR("MaxPool", 8, TranslateMaxPoolOp);
                REGISTER_OPERATOR("MaxPool", 10, TranslateMaxPoolOp);
                REGISTER_OPERATOR("MaxPool", 11, TranslateMaxPoolOp);
                REGISTER_OPERATOR("MaxPool", 12, TranslateMaxPoolOp);
                REGISTER_OPERATOR("Max", 1, TranslateMultiElementwiseOp<op::Maximum>);
                REGISTER_OPERATOR("Max", 6, TranslateMultiElementwiseOp<op::Maximum>);
                REGISTER_OPERATOR("Max", 8, TranslateMultiElementwiseOp<op::Maximum>);
                REGISTER_OPERATOR("Max", 12, TranslateMultiElementwiseOp<op::Maximum>);
                REGISTER_OPERATOR("Max", 13, TranslateMultiElementwiseOp<op::Maximum>);
                REGISTER_OPERATOR("Mean", 1, TranslateMeanOp);
                REGISTER_OPERATOR("Mean", 6, TranslateMeanOp);
                REGISTER_OPERATOR("Mean", 8, TranslateMeanOp);
                REGISTER_OPERATOR("Mean", 13, TranslateMeanOp);
                REGISTER_OPERATOR("MemcpyFromHost", 1, TranslateMemcpyFromHostOp);
                REGISTER_OPERATOR("MemcpyToHost", 1, TranslateMemcpyToHostOp);
                REGISTER_OPERATOR("Min", 1, TranslateMultiElementwiseOp<op::Minimum>);
                REGISTER_OPERATOR("Min", 6, TranslateMultiElementwiseOp<op::Minimum>);
                REGISTER_OPERATOR("Min", 8, TranslateMultiElementwiseOp<op::Minimum>);
                REGISTER_OPERATOR("Min", 12, TranslateMultiElementwiseOp<op::Minimum>);
                REGISTER_OPERATOR("Min", 13, TranslateMultiElementwiseOp<op::Minimum>);
                REGISTER_OPERATOR("Mul", 1, TranslateLegacyBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 6, TranslateLegacyBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 7, TranslateBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 13, TranslateBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 14, TranslateBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Neg", 1, TranslateUnaryOp<op::Negative>);
                REGISTER_OPERATOR("Neg", 6, TranslateUnaryOp<op::Negative>);
                REGISTER_OPERATOR("Neg", 13, TranslateUnaryOp<op::Negative>);
                REGISTER_OPERATOR("NonZero", 9, TranslateNonZeroOp);
                REGISTER_OPERATOR("NonZero", 13, TranslateNonZeroOp);
                REGISTER_OPERATOR("Not", 1, TranslateUnaryOp<op::Not>);
                REGISTER_OPERATOR("OneHot", 9, TranslateOneHotOp);
                REGISTER_OPERATOR("OneHot", 11, TranslateOneHotOp);
                REGISTER_OPERATOR("Or", 1, TranslateBinaryOp<op::Or>);
                REGISTER_OPERATOR("Or", 7, TranslateBinaryOp<op::Or>);
                REGISTER_OPERATOR("Pow", 1, TranslateBinaryOp<op::Power>);
                REGISTER_OPERATOR("Pow", 7, TranslateBinaryOp<op::Power>);
                REGISTER_OPERATOR("Pow", 12, TranslateBinaryOp<op::Power>);
                REGISTER_OPERATOR("Pow", 13, TranslateBinaryOp<op::Power>);
                REGISTER_OPERATOR("Pow", 15, TranslateBinaryOp<op::Power>);
                //REGISTER_OPERATOR("PRelu", 1, prelu);
                REGISTER_OPERATOR("Range", 11, TranslateRangeOp);
                REGISTER_OPERATOR("Reciprocal", 1, TranslateReciprocalOp);
                REGISTER_OPERATOR("Reciprocal", 6, TranslateReciprocalOp);
                REGISTER_OPERATOR("Reciprocal", 13, TranslateReciprocalOp);
                REGISTER_OPERATOR(
                    "ReduceL1", 1, PACK(TranslateReduceOp<op::Abs, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceL1", 11, PACK(TranslateReduceOp<op::Abs, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceL1", 13, PACK(TranslateReduceOp<op::Abs, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceL1", 18, PACK(TranslateReduceOp<op::Abs, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceL2", 1, PACK(TranslateReduceOp<op::Square, op::Sum, op::Sqrt>));
                REGISTER_OPERATOR(
                    "ReduceL2", 11, PACK(TranslateReduceOp<op::Square, op::Sum, op::Sqrt>));
                REGISTER_OPERATOR(
                    "ReduceL2", 13, PACK(TranslateReduceOp<op::Square, op::Sum, op::Sqrt>));
                REGISTER_OPERATOR(
                    "ReduceL2", 18, PACK(TranslateReduceOp<op::Square, op::Sum, op::Sqrt>));
                REGISTER_OPERATOR(
                    "ReduceLogSum", 1, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSum", 11, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSum", 13, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSum", 18, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSumExp", 1, PACK(TranslateReduceOp<op::Exp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSumExp", 11, PACK(TranslateReduceOp<op::Exp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSumExp", 13, PACK(TranslateReduceOp<op::Exp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceLogSumExp", 18, PACK(TranslateReduceOp<op::Exp, op::Sum, op::Log>));
                REGISTER_OPERATOR(
                    "ReduceMax", 1, PACK(TranslateReduceOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMax", 11, PACK(TranslateReduceOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMax", 12, PACK(TranslateReduceOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMax", 13, PACK(TranslateReduceOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMax", 18, PACK(TranslateReduceOp<op::NoOp, op::Max, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMean", 1, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Divide>));
                REGISTER_OPERATOR(
                    "ReduceMean", 11, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Divide>));
                REGISTER_OPERATOR(
                    "ReduceMean", 13, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Divide>));
                REGISTER_OPERATOR(
                    "ReduceMean", 18, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::Divide>));
                REGISTER_OPERATOR(
                    "ReduceMin", 1, PACK(TranslateReduceOp<op::NoOp, op::Min, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMin", 11, PACK(TranslateReduceOp<op::NoOp, op::Min, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMin", 12, PACK(TranslateReduceOp<op::NoOp, op::Min, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMin", 13, PACK(TranslateReduceOp<op::NoOp, op::Min, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceMin", 18, PACK(TranslateReduceOp<op::NoOp, op::Min, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceProd", 1, PACK(TranslateReduceOp<op::NoOp, op::Product, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceProd", 11, PACK(TranslateReduceOp<op::NoOp, op::Product, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceProd", 13, PACK(TranslateReduceOp<op::NoOp, op::Product, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceProd", 18, PACK(TranslateReduceOp<op::NoOp, op::Product, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceSum", 1, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceSum", 11, PACK(TranslateReduceOp<op::NoOp, op::Sum, op::NoOp>));
                REGISTER_OPERATOR("ReduceSum", 13, TranslateReduceSumOp);
                REGISTER_OPERATOR(
                    "ReduceSumSquare", 1, PACK(TranslateReduceOp<op::Square, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceSumSquare", 11, PACK(TranslateReduceOp<op::Square, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceSumSquare", 13, PACK(TranslateReduceOp<op::Square, op::Sum, op::NoOp>));
                REGISTER_OPERATOR(
                    "ReduceSumSquare", 18, PACK(TranslateReduceOp<op::Square, op::Sum, op::NoOp>));
                REGISTER_OPERATOR("Relu", 1, TranslateUnaryOp<op::Relu>);
                REGISTER_OPERATOR("Reshape", 1, TranslateReshapeOp);
                REGISTER_OPERATOR("ReshapeGrad", 1, TranslateReshapeGradOp);
                //REGISTER_OPERATOR("Selu", 1, selu);
                REGISTER_OPERATOR("Shape", 1, TranslateShapeOp);
                REGISTER_OPERATOR("Shape", 15, TranslateShapeOp);
                REGISTER_OPERATOR("Sigmoid", 1, TranslateUnaryOp<op::Sigmoid>);
                REGISTER_OPERATOR("Sin", 1, TranslateUnaryOp<op::Sin>);
                REGISTER_OPERATOR("Slice", 1, TranslateSliceOp);
                REGISTER_OPERATOR("Slice", 10, TranslateSliceOp);
                REGISTER_OPERATOR("Size", 1, TranslateSizeOp);
                REGISTER_OPERATOR("Size", 13, TranslateSizeOp);
                REGISTER_DOMAIN_OPERATOR(
                    "com.microsoft", "SkipLayerNormalization", 1, TranslateSkipLayerNormOp);
                REGISTER_OPERATOR("Softmax", 1, TranslateSoftmaxOp);
                REGISTER_OPERATOR(
                    "SoftmaxCrossEntropyLoss", 1, TranslateSparseSoftmaxCrossEntropyOp);
                REGISTER_OPERATOR("SoftmaxGrad", 1, TranslateSoftmaxGradOp);
                REGISTER_OPERATOR(
                    "SparseSoftmaxCrossEntropy", 1, TranslateSparseSoftmaxCrossEntropyOp);
                REGISTER_OPERATOR(
                    "SparseSoftmaxCrossEntropyGrad", 1, TranslateSparseSoftmaxCrossEntropyGradOp);
                //REGISTER_OPERATOR("Softplus", 1, softplus);
                //REGISTER_OPERATOR("Softsign", 1, softsign);
                REGISTER_OPERATOR("Split", 1, TranslateSplitOp);
                REGISTER_OPERATOR("Sqrt", 1, TranslateUnaryOp<op::Sqrt>);
                REGISTER_OPERATOR("Sqrt", 6, TranslateUnaryOp<op::Sqrt>);
                REGISTER_OPERATOR("Sqrt", 13, TranslateUnaryOp<op::Sqrt>);
                REGISTER_OPERATOR("Squeeze", 1, TranslateSqueezeOp);
                REGISTER_OPERATOR("Pad", 1, TranslatePadOp);
                REGISTER_OPERATOR("Squeeze", 11, TranslateSqueezeOp);
                REGISTER_OPERATOR("Sub", 1, TranslateLegacyBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sub", 6, TranslateBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sub", 7, TranslateBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sum", 1, TranslateMultiElementwiseOp<op::Add>);
                REGISTER_OPERATOR("Sum", 6, TranslateMultiElementwiseOp<op::Add>);
                REGISTER_OPERATOR("Sum", 8, TranslateMultiElementwiseOp<op::Add>);
                REGISTER_OPERATOR("Sum", 13, TranslateMultiElementwiseOp<op::Add>);
                REGISTER_OPERATOR("Tan", 1, TranslateUnaryOp<op::Tan>);
                REGISTER_OPERATOR("Tanh", 1, TranslateUnaryOp<op::Tanh>);
                REGISTER_OPERATOR("Tanh", 6, TranslateUnaryOp<op::Tanh>);
                REGISTER_OPERATOR("Tanh", 13, TranslateUnaryOp<op::Tanh>);
                REGISTER_OPERATOR("TanhGrad", 1, TranslateTanhGradOp);
                REGISTER_OPERATOR("Tile", 6, TranslateTileOp);
                REGISTER_OPERATOR("Tile", 13, TranslateTileOp);
                // REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
                REGISTER_OPERATOR("TrainableDropout", 1, TranslateTrainableDropoutOp);
                REGISTER_DOMAIN_OPERATOR(
                    "com.microsoft", "TrainableDropoutGrad", 1, TranslateTrainableDropoutGradOp);
                REGISTER_OPERATOR("Transpose", 1, TranslateTransposeOp);
                REGISTER_OPERATOR("Transpose", 13, TranslateTransposeOp);
                REGISTER_DOMAIN_OPERATOR("com.microsoft", "TransposeMatMul", 1, TranslateMatmulOp);
                REGISTER_OPERATOR("Unsqueeze", 1, TranslateUnsqueezeOp);
                REGISTER_OPERATOR("Unsqueeze", 13, TranslateUnsqueezeOp);
                REGISTER_OPERATOR("ConvTranspose", 1, TranslateConvTransposeOp);
                REGISTER_OPERATOR("ConvTranspose", 11, TranslateConvTransposeOp);
                REGISTER_OPERATOR("Resize", 1, TranslateResizeOp);
                REGISTER_OPERATOR("Upsample", 1, TranslateResizeOp);
                REGISTER_OPERATOR("Where", 9, TranslateWhereOp);
                REGISTER_OPERATOR("Where", 16, TranslateWhereOp);
                REGISTER_OPERATOR("ScatterND", 11, TranslateScatterNDOp);
                REGISTER_OPERATOR("DepthToSpace", 1, TranslateDepthToSpaceOp);
                REGISTER_OPERATOR("DepthToSpace", 11, TranslateDepthToSpaceOp);
                REGISTER_OPERATOR("DepthToSpace", 13, TranslateDepthToSpaceOp);
                REGISTER_DOMAIN_OPERATOR("org.pytorch.aten", "roll", 1, TranslateRollOp);
                // REGISTER_OPERATOR("Xor", 1, logical_xor);
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
