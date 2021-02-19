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
#include "op/binaryop.hpp"
#include "op/cast.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/const_of_shape.hpp"
#include "op/constant.hpp"
#include "op/conv.hpp"
#include "op/conv_trans.hpp"
#include "op/div_grad.hpp"
#include "op/dropout.hpp"
#include "op/embed_layer_norm.hpp"
#include "op/erf_grad.hpp"
#include "op/expand.hpp"
#include "op/flatten.hpp"
#include "op/gather.hpp"
#include "op/gemm.hpp"
#include "op/identity.hpp"
#include "op/index_reduce.hpp"
#include "op/layer_norm.hpp"
#include "op/leaky_relu.hpp"
#include "op/log_softmax.hpp"
#include "op/lstm.hpp"
#include "op/matmul.hpp"
#include "op/memory_copy.hpp"
#include "op/non_zero.hpp"
#include "op/one_hot.hpp"
#include "op/pool.hpp"
#include "op/reduce.hpp"
#include "op/reshape.hpp"
#include "op/resize.hpp"
#include "op/shape.hpp"
#include "op/skip_layer_norm.hpp"
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
                    int64_t avail_version = version;
                    while (avail_version > 0)
                    {
                        const auto it = map.find(avail_version--);
                        if (it != std::end(map))
                        {
                            return it->second;
                        }
                    }
                    NNFUSION_CHECK_FAIL()
                        << "Unsupported version: " << (domain.empty() ? "" : domain + ".") << name
                        << ":" << std::to_string(version);
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
                NNFUSION_CHECK(dm != std::end(m_map)) << "Unknown Domain: " << domain;

                for (const auto& op : dm->second)
                {
                    result.emplace(op.first, detail::find(op.first, version, domain, op.second));
                }
                return result;
            }

#define REGISTER_DOMAIN_OPERATOR(domain_, name_, ver_, fn_)                                        \
    m_map[domain_][name_].emplace(                                                                 \
        ver_,                                                                                      \
        std::bind(                                                                                 \
            set_##ver_::fn_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3))

#define REGISTER_OPERATOR(name_, ver_, fn_) REGISTER_DOMAIN_OPERATOR("", name_, ver_, fn_)

#define REGISTER_EMPTY_DOMAIN(domain_) m_map[domain_]

            OperatorsBridge::OperatorsBridge()
            {
                REGISTER_EMPTY_DOMAIN("com.microsoft.nchwc");
                REGISTER_EMPTY_DOMAIN("ai.onnx.training");
                REGISTER_EMPTY_DOMAIN("ai.onnx.ml");
                REGISTER_EMPTY_DOMAIN("ai.onnx.preview.training");
                REGISTER_EMPTY_DOMAIN("com.microsoft");
                REGISTER_EMPTY_DOMAIN("com.microsoft.mlfeaturizers");
                REGISTER_OPERATOR("Abs", 1, TranslateUnaryOp<op::Abs>);
                REGISTER_OPERATOR("Acos", 1, TranslateUnaryOp<op::Acos>);
                REGISTER_OPERATOR("AdamOptimizer", 1, TranslateAdamOptimizerOp);
                REGISTER_OPERATOR("Add", 1, TranslateLegacyBinaryOp<op::Add>);
                REGISTER_OPERATOR("Add", 7, TranslateBinaryOp<op::Add>);
                REGISTER_OPERATOR("And", 1, TranslateBinaryOp<op::And>);
                REGISTER_OPERATOR("ArgMin", 1, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMax", 1, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("Asin", 1, TranslateUnaryOp<op::Asin>);
                REGISTER_OPERATOR("Atan", 1, TranslateUnaryOp<op::Atan>);
                REGISTER_DOMAIN_OPERATOR("com.microsoft", "Attention", 1, TranslateAttentionOp);
                REGISTER_OPERATOR("AveragePool", 1, TranslatePoolOp<op::AvgPool>);
                REGISTER_OPERATOR("BatchNormalization", 1, TranslateBatchNormOp);
                REGISTER_OPERATOR("Cast", 1, TranslateCastOp);
                REGISTER_OPERATOR("Ceil", 1, TranslateUnaryOp<op::Ceiling>);
                REGISTER_OPERATOR("Clip", 1, TranslateClipOp);
                REGISTER_OPERATOR("Concat", 1, TranslateConcatOp);
                REGISTER_OPERATOR("Constant", 1, TranslateConstantOp);
                REGISTER_OPERATOR("ConstantOfShape", 1, TranslateConstantOfShapeOp);
                REGISTER_OPERATOR("Conv", 1, TranslateConvOp);
                REGISTER_OPERATOR("Cos", 1, TranslateUnaryOp<op::Cos>);
                REGISTER_OPERATOR("Div", 1, TranslateLegacyBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 7, TranslateBinaryOp<op::Divide>);
                REGISTER_OPERATOR("DivGrad", 1, TranslateDivGradOp);
                REGISTER_OPERATOR("Dropout", 1, TranslateDropoutOp);
                //REGISTER_OPERATOR("Elu", 1, elu);
                REGISTER_DOMAIN_OPERATOR(
                    "com.microsoft", "EmbedLayerNormalization", 1, TranslateEmbedLayerNormOp);
                REGISTER_OPERATOR("Equal", 1, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Erf", 1, TranslateUnaryOp<op::Erf>);
                REGISTER_OPERATOR("ErfGrad", 1, TranslateErfGradOp);
                REGISTER_OPERATOR("Exp", 1, TranslateUnaryOp<op::Exp>);
                REGISTER_OPERATOR("Expand", 1, TranslateExpandOp);
                REGISTER_OPERATOR("Flatten", 1, TranslateFlattenOp);
                REGISTER_OPERATOR("Floor", 1, TranslateUnaryOp<op::Floor>);
                REGISTER_OPERATOR("Gather", 1, TranslateGatherOp);
                REGISTER_OPERATOR("GatherGrad", 1, TranslateGatherGradOp);
                REGISTER_OPERATOR(
                    "GatherND", 1, TranslateGatherNDOp); // actually it's available since opset_11
                REGISTER_OPERATOR("GatherNDGrad", 1, TranslateGatherNDGradOp);
                REGISTER_OPERATOR("Gelu", 1, TranslateUnaryOp<op::Gelu>);
                REGISTER_OPERATOR("Gemm", 1, TranslateGemmOp);
                REGISTER_OPERATOR("GlobalAveragePool", 1, TranslatePoolOp<op::AvgPool>);
                REGISTER_OPERATOR("GlobalMaxPool", 1, TranslatePoolOp<op::MaxPool>);
                REGISTER_OPERATOR("Greater", 1, TranslateBinaryOp<op::Greater>);
                //REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
                REGISTER_OPERATOR("Identity", 1, TranslateIdentityOp);
                REGISTER_OPERATOR("LayerNormalization", 1, TranslateLayerNormalizationOp);
                REGISTER_OPERATOR("LayerNormalizationGrad", 1, TranslateLayerNormalizationGradOp);
                REGISTER_OPERATOR("LeakyRelu", 1, TranslateLeakyReluOp);
                REGISTER_OPERATOR("Less", 1, TranslateBinaryOp<op::Less>);
                REGISTER_OPERATOR("Log", 1, TranslateUnaryOp<op::Log>);
                REGISTER_OPERATOR("LogSoftmax", 1, TranslateLogSoftmaxOp);
                //REGISTER_OPERATOR("LRN", 1, lrn);
                REGISTER_OPERATOR("LSTM", 1, TranslateLstmOp);
                REGISTER_OPERATOR("MatMul", 1, TranslateMatmulOp);
                REGISTER_OPERATOR("MaxPool", 1, TranslatePoolOp<op::MaxPool>);
                REGISTER_OPERATOR("Max", 1, TranslateLegacyBinaryOp<op::Maximum>);
                //REGISTER_OPERATOR("Mean", 1, mean);
                REGISTER_OPERATOR("MemcpyFromHost", 1, TranslateMemcpyFromHostOp);
                REGISTER_OPERATOR("MemcpyToHost", 1, TranslateMemcpyToHostOp);
                REGISTER_OPERATOR("Min", 1, TranslateLegacyBinaryOp<op::Minimum>);
                REGISTER_OPERATOR("Mul", 1, TranslateLegacyBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 7, TranslateBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Neg", 1, TranslateUnaryOp<op::Negative>);
                REGISTER_OPERATOR("NonZero", 1, TranslateNonZeroOp);
                REGISTER_OPERATOR("Not", 1, TranslateUnaryOp<op::Not>);
                REGISTER_OPERATOR("OneHot", 1, TranslateOneHotOp);
                REGISTER_OPERATOR("Or", 1, TranslateBinaryOp<op::Or>);
                REGISTER_OPERATOR("Pow", 1, TranslateBinaryOp<op::Power>);
                //REGISTER_OPERATOR("PRelu", 1, prelu);
                //REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
                //REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
                //REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
                //REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
                //REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
                //REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
                REGISTER_OPERATOR("ReduceMean", 1, TranslateReduceMeanOp);
                //REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
                //REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
                REGISTER_OPERATOR("ReduceSum", 1, TranslateReduceSumOp);
                //REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
                REGISTER_OPERATOR("Relu", 1, TranslateUnaryOp<op::Relu>);
                REGISTER_OPERATOR("Reshape", 1, TranslateReshapeOp);
                REGISTER_OPERATOR("ReshapeGrad", 1, TranslateReshapeGradOp);
                //REGISTER_OPERATOR("Selu", 1, selu);
                REGISTER_OPERATOR("Shape", 1, TranslateShapeOp);
                REGISTER_OPERATOR("Sigmoid", 1, TranslateUnaryOp<op::Sigmoid>);
                REGISTER_OPERATOR("Sin", 1, TranslateUnaryOp<op::Sin>);
                REGISTER_OPERATOR("Slice", 1, TranslateSliceOp);
                REGISTER_OPERATOR("Slice", 10, TranslateSliceOp);
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
                REGISTER_OPERATOR("Squeeze", 1, TranslateSqueezeOp);
                REGISTER_OPERATOR("Squeeze", 11, TranslateSqueezeOp);
                REGISTER_OPERATOR("Sub", 1, TranslateLegacyBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sub", 7, TranslateBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sum", 1, TranslateSumOp);
                REGISTER_OPERATOR("Tan", 1, TranslateUnaryOp<op::Tan>);
                REGISTER_OPERATOR("Tanh", 1, TranslateUnaryOp<op::Tanh>);
                REGISTER_OPERATOR("TanhGrad", 1, TranslateTanhGradOp);
                REGISTER_OPERATOR("Tile", 1, TranslateTileOp);
                // REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
                REGISTER_OPERATOR("TrainableDropout", 1, TranslateTrainableDropoutOp);
                REGISTER_DOMAIN_OPERATOR(
                    "com.microsoft", "TrainableDropoutGrad", 1, TranslateTrainableDropoutGradOp);
                REGISTER_OPERATOR("Transpose", 1, TranslateTransposeOp);
                REGISTER_DOMAIN_OPERATOR("com.microsoft", "TransposeMatMul", 1, TranslateMatmulOp);
                REGISTER_OPERATOR("Unsqueeze", 1, TranslateUnsqueezeOp);
                REGISTER_OPERATOR("ConvTranspose", 1, TranslateConvTransposeOp);
                REGISTER_OPERATOR("Resize", 1, TranslateResizeOp);
                REGISTER_OPERATOR("Upsample", 1, TranslateResizeOp);
                REGISTER_OPERATOR("Where", 1, TranslateWhereOp);
                // REGISTER_OPERATOR("Xor", 1, logical_xor);
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
