// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "batchnorm_inference_folding_pass.hpp"
#include <queue>
#include "nnfusion/core/kernels/cuda_gpu/kernels/anyop.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/anyop.hpp"
#include "nnfusion/core/operators/op_define/add.hpp"
#include "nnfusion/core/operators/op_define/batch_norm.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/convolution.hpp"
#include "nnfusion/core/operators/op_define/dot.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "runtime_const_folding_pass.hpp"

DEFINE_bool(fbatchnorm_inference_folding,
            true,
            "BatchNormInference folding for accelerating model inference");
DECLARE_string(fconst_folding_backend);

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

const static std::vector<std::vector<std::string>> BN_FOLDING_PATTERNS = {
    {"Convolution", "Broadcast", "Add", "BatchNormInference"},
    {"Convolution", "Add", "BatchNormInference"},
    {"Convolution", "BatchNormInference"}};

namespace
{
    struct TaggedNode
    {
        TaggedNode()
            : node(nullptr)
            , ready_inputs(0)
            , visited(false)
        {
        }

        std::shared_ptr<GNode> node;
        size_t ready_inputs;
        bool visited;
    };
} // namespace

class BatchNormInferenceOptimizer
{
public:
    BatchNormInferenceOptimizer(std::shared_ptr<Graph> g, std::vector<std::string> p)
        : m_graph(g)
        , m_pattern(p)
    {
        m_nodes.resize(m_graph->get_max_node_id());
    }

    void MatchAndFolding()
    {
        std::queue<size_t> ready;
        for (auto node : m_graph->get_ordered_ops())
        {
            size_t id = node->get_id();
            m_nodes[id] = std::make_shared<TaggedNode>();
            m_nodes[id]->node = node;
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
            {
                ready.push(id);
            }
        }

        while (!ready.empty())
        {
            size_t node_id = ready.front();
            ready.pop();

            node_id = MatchPattern(node_id);
            auto tn = m_nodes[node_id];
            tn->visited = true;
            // std::cout << tn->node->get_name() << std::endl;
            for (auto edge : tn->node->get_out_edges())
            {
                auto dst = m_nodes[edge->get_dst()->get_id()];
                // node that will not be computed
                if (!dst)
                    continue;
                dst->ready_inputs++;
                NNFUSION_CHECK(!(dst->visited));
                if (dst->ready_inputs >= dst->node->get_in_edges().size())
                {
                    NNFUSION_CHECK(dst->ready_inputs == dst->node->get_in_edges().size());
                    ready.push(dst->node->get_id());
                }
            }
        }
    }

private:
    size_t MatchPattern(size_t id)
    {
        std::shared_ptr<TaggedNode> tn = m_nodes[id];
        std::vector<std::shared_ptr<TaggedNode>> matched;

        for (size_t i = 0; i < m_pattern.size(); i++)
        {
            if (tn && tn->node->get_op_type() == m_pattern[i])
            {
                if (i == m_pattern.size() - 1 || tn->node->get_out_edges().size() == 1)
                {
                    matched.push_back(tn);
                    for (auto edge : tn->node->get_out_edges())
                    {
                        tn = m_nodes[edge->get_dst()->get_id()];
                    }
                    continue;
                }
            }
            return id;
        }

        // verify matched pattern
        for (auto i = 0; i < m_pattern.size(); i++)
        {
            if (matched[i]->node->get_op_type() != m_pattern[i])
            {
                return id;
            }
        }

        std::string identifier = "";
        for (auto m_tn : matched)
        {
            std::shared_ptr<KernelContext> ctx(new KernelContext(m_tn->node));
            identifier += ctx->generate_identifier();
        }

        NNFUSION_LOG(INFO) << "BatchNormInference folding pattern found: " << identifier;
        auto ret_id = id;
        auto folding_id = Folding(matched, identifier);
        if (folding_id >= 0)
        {
            NNFUSION_LOG(INFO) << "BatchNormInference folding applied: " << identifier;
            ret_id = (size_t)folding_id;
        }
        else
        {
            NNFUSION_LOG(INFO) << "BatchNormInference folding failed: " << identifier
                               << ", fallback.";
        }

        return ret_id;
    }

    std::vector<double> ExtractConstantData(const std::shared_ptr<op::Constant> ptr,
                                            const element::Type& dtype)
    {
        if (dtype == element::f64)
        {
            auto data_vec = ptr->get_vector<double>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::f32)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::bf16)
        {
            auto data_vec = bfloat16::to_float_vector(ptr->get_vector<bfloat16>());
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i64)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i32)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i16)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }
        else if (dtype == element::i8)
        {
            auto data_vec = ptr->get_vector<float>();
            std::vector<double> ret;
            for (auto i = 0; i < data_vec.size(); i++)
            {
                ret.push_back((double)data_vec[i]);
            }
            return ret;
        }

        NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
        std::vector<double> ret;
        return ret;
    }

    int64_t Folding(const std::vector<std::shared_ptr<TaggedNode>>& matched,
                    const std::string& identifier)
    {
        if (m_pattern ==
            std::vector<std::string>({"Convolution", "Broadcast", "Add", "BatchNormInference"}))
        {
            auto conv_node = matched[0]->node;
            auto conv_node_op_ptr =
                std::dynamic_pointer_cast<op::Convolution>(conv_node->get_op_ptr());
            auto broadcast_node = matched[1]->node;
            auto add_node = matched[2]->node;
            auto bn_node = matched[3]->node;
            auto bn_node_op_ptr =
                std::dynamic_pointer_cast<op::BatchNormInference>(bn_node->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(conv_node_op_ptr);
            NNFUSION_CHECK_NOT_NULLPTR(bn_node_op_ptr);

            auto conv_weight_node = conv_node->get_in_edge(1)->get_src();
            auto conv_weight_const_ptr =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(conv_weight_node->get_op_ptr());
            if (conv_weight_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "Convolution weight is not constant. Runtime_const_folding pass may solve "
                       "this problem. Fallback.";
                return -1;
            }
            auto conv_bias_node = broadcast_node->get_in_edge(0)->get_src();
            auto conv_bias_const_ptr =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(conv_bias_node->get_op_ptr());
            if (conv_bias_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "Convolution bias is not constant. Runtime_const_folding pass may solve "
                       "this problem. Fallback.";
                return -1;
            }

            auto bn_gain_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(0)->get_src()->get_op_ptr());
            auto bn_bias_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(1)->get_src()->get_op_ptr());
            auto bn_mean_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(3)->get_src()->get_op_ptr());
            auto bn_variance_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(4)->get_src()->get_op_ptr());
            if (bn_gain_const_ptr == nullptr || bn_bias_const_ptr == nullptr ||
                bn_mean_const_ptr == nullptr || bn_variance_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "BatchNormInference weight is not constant. "
                                                  "Runtime_const_folding pass may solve "
                                                  "this problem. Fallback.";
                return -1;
            }

            auto dtype = conv_node->get_input_element_type(1);

            if (!(dtype == element::f64 || dtype == element::f32 || dtype == element::bf16 ||
                  dtype == element::i64 || dtype == element::i32 || dtype == element::i16 ||
                  dtype == element::i8))
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Not support DataType " << dtype << ". Fallback.";
                return -1;
            }

            std::vector<double> conv_weight = ExtractConstantData(conv_weight_const_ptr, dtype);
            std::vector<double> conv_bias = ExtractConstantData(conv_bias_const_ptr, dtype);
            std::vector<double> bn_gain = ExtractConstantData(bn_gain_const_ptr, dtype);
            std::vector<double> bn_bias = ExtractConstantData(bn_bias_const_ptr, dtype);
            std::vector<double> bn_mean = ExtractConstantData(bn_mean_const_ptr, dtype);
            std::vector<double> bn_variance = ExtractConstantData(bn_variance_const_ptr, dtype);
            double bn_epsilon = bn_node_op_ptr->get_eps_value();

            // compute and replace constant data
            std::shared_ptr<op::Constant> new_conv_weight_op_ptr;
            std::shared_ptr<op::Constant> new_conv_bias_op_ptr;
            if (dtype == element::f64)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<double>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<double>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::f32)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<float>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::bf16)
            {
                auto conv_weight_converted = bfloat16::from_float_vector(
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon));
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted =
                    bfloat16::from_float_vector(ConvertConvolutionBias<float>(
                        conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon));
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::i64)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int64_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int64_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::i32)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int32_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int32_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::i16)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int16_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int16_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else if (dtype == element::i8)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<int8_t>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int8_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, broadcast_node->get_input_shape(0), conv_bias_converted.data());
            }
            else
            {
                NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
            }
            auto new_conv_weight_gnode =
                std::make_shared<nnfusion::graph::GNode>(new_conv_weight_op_ptr, GNodeVector());
            m_graph->replace_node(conv_weight_node, new_conv_weight_gnode, false);
            auto new_conv_bias_gnode =
                std::make_shared<nnfusion::graph::GNode>(new_conv_bias_op_ptr, GNodeVector());
            m_graph->replace_node(conv_bias_node, new_conv_bias_gnode, false);

            // delete batchnorm_inference node
            auto bn_out_edges = bn_node->get_out_edges();
            int next_output_id = 0;
            for (auto bn_out_edge : bn_out_edges)
            {
                auto output_id =
                    bn_out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id++;
                if (output_id != Graph::kControlSlot && output_id == 0)
                {
                    add_node->set_output(output_id,
                                         bn_node->get_outputs().at(bn_out_edge->get_src_output()));
                }
                m_graph->add_edge(
                    add_node, 0, bn_out_edge->get_dst(), bn_out_edge->get_dst_input());
            }
            m_graph->remove_node(bn_node);

            return add_node->get_id();
        }
        else if (m_pattern ==
                 std::vector<std::string>({"Convolution", "Add", "BatchNormInference"}))
        {
            auto conv_node = matched[0]->node;
            auto conv_node_op_ptr =
                std::dynamic_pointer_cast<op::Convolution>(conv_node->get_op_ptr());
            auto add_node = matched[1]->node;
            auto bn_node = matched[2]->node;
            auto bn_node_op_ptr =
                std::dynamic_pointer_cast<op::BatchNormInference>(bn_node->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(conv_node_op_ptr);
            NNFUSION_CHECK_NOT_NULLPTR(bn_node_op_ptr);
            auto conv_weight_node = conv_node->get_in_edge(1)->get_src();
            auto conv_weight_const_ptr =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(conv_weight_node->get_op_ptr());
            if (conv_weight_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "Convolution weight is not constant. Runtime_const_folding pass may solve "
                       "this problem. Fallback.";
                return -1;
            }
            auto conv_bias_node = add_node->get_in_edge(1)->get_src();
            auto conv_bias_const_ptr =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(conv_bias_node->get_op_ptr());
            if (conv_bias_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "Convolution bias is not constant. Runtime_const_folding pass may solve "
                       "this problem. Fallback.";
                return -1;
            }

            auto bn_gain_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(0)->get_src()->get_op_ptr());
            auto bn_bias_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(1)->get_src()->get_op_ptr());
            auto bn_mean_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(3)->get_src()->get_op_ptr());
            auto bn_variance_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(4)->get_src()->get_op_ptr());
            if (bn_gain_const_ptr == nullptr || bn_bias_const_ptr == nullptr ||
                bn_mean_const_ptr == nullptr || bn_variance_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "BatchNormInference weight is not constant. "
                                                  "Runtime_const_folding pass may solve "
                                                  "this problem. Fallback.";
                return -1;
            }

            auto dtype = conv_node->get_input_element_type(1);

            if (!(dtype == element::f64 || dtype == element::f32 || dtype == element::bf16 ||
                  dtype == element::i64 || dtype == element::i32 || dtype == element::i16 ||
                  dtype == element::i8))
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Not support DataType " << dtype << ". Fallback.";
                return -1;
            }

            std::vector<double> conv_weight = ExtractConstantData(conv_weight_const_ptr, dtype);
            std::vector<double> conv_bias;
            std::vector<double> bn_gain = ExtractConstantData(bn_gain_const_ptr, dtype);
            std::vector<double> bn_bias = ExtractConstantData(bn_bias_const_ptr, dtype);
            std::vector<double> bn_mean = ExtractConstantData(bn_mean_const_ptr, dtype);
            std::vector<double> bn_variance = ExtractConstantData(bn_variance_const_ptr, dtype);
            double bn_epsilon = bn_node_op_ptr->get_eps_value();

            std::vector<double> conv_bias_tmp = ExtractConstantData(conv_bias_const_ptr, dtype);
            auto conv_output_shape = conv_node->get_output_shape(0);
            bool is_nhwc = (conv_node_op_ptr->get_data_format() == "NHWC");
            if (is_nhwc)
            {
                for (auto c = 0; c < bn_bias.size(); c++)
                {
                    conv_bias.push_back(conv_bias_tmp[c]);
                }
            }
            else
            {
                for (auto c = 0; c < bn_bias.size(); c++)
                {
                    conv_bias.push_back(
                        conv_bias_tmp[c * conv_output_shape[2] * conv_output_shape[3]]);
                }
            }

            // compute and replace constant data
            std::shared_ptr<op::Constant> new_conv_weight_op_ptr;
            std::shared_ptr<op::Constant> new_conv_bias_op_ptr;
            if (dtype == element::f64)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<double>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<double>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::f32)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<float>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::bf16)
            {
                auto conv_weight_converted = bfloat16::from_float_vector(
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon));
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted =
                    bfloat16::from_float_vector(ConvertConvolutionBias<float>(
                        conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon));
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::i64)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int64_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int64_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::i32)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int32_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int32_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::i16)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int16_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int16_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else if (dtype == element::i8)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<int8_t>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
                auto conv_bias_converted = ConvertConvolutionBias<int8_t>(
                    conv_bias, bn_gain, bn_bias, bn_mean, bn_variance, bn_epsilon);
                new_conv_bias_op_ptr = std::make_shared<op::Constant>(
                    dtype, bn_node->get_input_shape(1), conv_bias_converted.data());
            }
            else
            {
                NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
            }
            auto new_conv_weight_gnode =
                std::make_shared<nnfusion::graph::GNode>(new_conv_weight_op_ptr, GNodeVector());
            m_graph->replace_node(conv_weight_node, new_conv_weight_gnode, false);
            auto new_conv_bias_gnode =
                std::make_shared<nnfusion::graph::GNode>(new_conv_bias_op_ptr, GNodeVector());
            new_conv_bias_gnode->set_output_type_and_shape(
                0, dtype, bn_node->get_input_partial_shape(1));
            m_graph->add_node(new_conv_bias_gnode);
            m_nodes.resize(m_graph->get_max_node_id());
            m_nodes[new_conv_bias_gnode->get_id()] = std::make_shared<TaggedNode>();
            m_nodes[new_conv_bias_gnode->get_id()]->node = new_conv_bias_gnode;

            // BiasAdd: broadcast -> add
            nnfusion::AxisSet broadcast_axes;
            if (is_nhwc)
            {
                for (size_t i = 0; i < conv_output_shape.size() - 1; i++)
                {
                    broadcast_axes.insert(i);
                }
            }
            else
            {
                for (size_t i = 0; i < conv_output_shape.size(); i++)
                {
                    if (i != 1)
                    {
                        broadcast_axes.insert(i);
                    }
                }
            }
            auto new_broadcast_gnode = m_graph->add_node_and_edge(
                std::make_shared<op::Broadcast>(conv_output_shape, broadcast_axes),
                {new_conv_bias_gnode});
            // this fix is for a fair baseline on performance evaluation
            shared_ptr<KernelContext> ke_ctx(new KernelContext(new_broadcast_gnode));
            KernelEmitter::Pointer any_op_ke =
                std::make_shared<nnfusion::kernels::cuda::AnyOP>(ke_ctx);
            any_op_ke->get_or_emit_source();
            (*new_broadcast_gnode)["Kernel_Selection_Result"] =
                std::make_pair(NNFusion_DeviceType::CUDA_GPU, any_op_ke);

            m_nodes.resize(m_graph->get_max_node_id());
            m_nodes[new_broadcast_gnode->get_id()] = std::make_shared<TaggedNode>();
            m_nodes[new_broadcast_gnode->get_id()]->node = new_broadcast_gnode;
            auto new_add_gnode = m_graph->add_node_and_edge(
                std::make_shared<op::Add>(),
                {add_node->get_in_edge(0)->get_src(), new_broadcast_gnode});
            m_nodes.resize(m_graph->get_max_node_id());
            m_nodes[new_add_gnode->get_id()] = std::make_shared<TaggedNode>();
            m_nodes[new_add_gnode->get_id()]->node = new_add_gnode;
            m_graph->remove_node(add_node->get_in_edge(1)->get_src());
            m_graph->remove_node(add_node);

            // delete batchnorm_inference node
            auto bn_out_edges = bn_node->get_out_edges();
            int next_output_id = 0;
            for (auto bn_out_edge : bn_out_edges)
            {
                auto output_id =
                    bn_out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id++;
                if (output_id != Graph::kControlSlot && output_id == 0)
                {
                    new_add_gnode->set_output(
                        output_id, bn_node->get_outputs().at(bn_out_edge->get_src_output()));
                }
                m_graph->add_edge(
                    new_add_gnode, 0, bn_out_edge->get_dst(), bn_out_edge->get_dst_input());
            }
            m_graph->remove_node(bn_node);

            return new_add_gnode->get_id();
        }
        else if (m_pattern == std::vector<std::string>({"Convolution", "BatchNormInference"}))
        {
            auto conv_node = matched[0]->node;
            auto conv_node_op_ptr =
                std::dynamic_pointer_cast<op::Convolution>(conv_node->get_op_ptr());
            auto bn_node = matched[1]->node;
            auto bn_node_op_ptr =
                std::dynamic_pointer_cast<op::BatchNormInference>(bn_node->get_op_ptr());
            NNFUSION_CHECK_NOT_NULLPTR(conv_node_op_ptr);
            NNFUSION_CHECK_NOT_NULLPTR(bn_node_op_ptr);
            auto conv_weight_node = conv_node->get_in_edge(1)->get_src();
            auto conv_weight_const_ptr =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(conv_weight_node->get_op_ptr());
            if (conv_weight_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING)
                    << "Convolution weight is not constant. Runtime_const_folding pass may solve "
                       "this problem. Fallback.";
                return -1;
            }

            auto bn_gain_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(0)->get_src()->get_op_ptr());
            auto bn_bias_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(1)->get_src()->get_op_ptr());
            auto bn_mean_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(3)->get_src()->get_op_ptr());
            auto bn_variance_const_ptr = std::dynamic_pointer_cast<op::Constant>(
                bn_node->get_in_edge(4)->get_src()->get_op_ptr());
            if (bn_gain_const_ptr == nullptr || bn_bias_const_ptr == nullptr ||
                bn_mean_const_ptr == nullptr || bn_variance_const_ptr == nullptr)
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "BatchNormInference weight is not constant. "
                                                  "Runtime_const_folding pass may solve "
                                                  "this problem. Fallback.";
                return -1;
            }

            auto dtype = conv_node->get_input_element_type(1);

            if (!(dtype == element::f64 || dtype == element::f32 || dtype == element::bf16 ||
                  dtype == element::i64 || dtype == element::i32 || dtype == element::i16 ||
                  dtype == element::i8))
            {
                NNFUSION_LOG(NNFUSION_WARNING) << "Not support DataType " << dtype << ". Fallback.";
                return -1;
            }

            std::vector<double> conv_weight = ExtractConstantData(conv_weight_const_ptr, dtype);
            std::vector<double> bn_gain = ExtractConstantData(bn_gain_const_ptr, dtype);
            std::vector<double> bn_bias = ExtractConstantData(bn_bias_const_ptr, dtype);
            std::vector<double> bn_mean = ExtractConstantData(bn_mean_const_ptr, dtype);
            std::vector<double> bn_variance = ExtractConstantData(bn_variance_const_ptr, dtype);
            double bn_epsilon = bn_node_op_ptr->get_eps_value();

            // compute and replace constant data
            std::shared_ptr<op::Constant> new_conv_weight_op_ptr;
            if (dtype == element::f64)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<double>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::f32)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::bf16)
            {
                auto conv_weight_converted = bfloat16::from_float_vector(
                    ConvertConvolutionWeight<float>(conv_weight, bn_gain, bn_variance, bn_epsilon));
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::i64)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int64_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::i32)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int32_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::i16)
            {
                auto conv_weight_converted = ConvertConvolutionWeight<int16_t>(
                    conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else if (dtype == element::i8)
            {
                auto conv_weight_converted =
                    ConvertConvolutionWeight<int8_t>(conv_weight, bn_gain, bn_variance, bn_epsilon);
                new_conv_weight_op_ptr = std::make_shared<op::Constant>(
                    dtype, conv_node->get_input_shape(1), conv_weight_converted.data());
            }
            else
            {
                NNFUSION_CHECK_FAIL() << "Not support DataType " << dtype;
            }
            auto new_conv_weight_gnode =
                std::make_shared<nnfusion::graph::GNode>(new_conv_weight_op_ptr, GNodeVector());
            m_graph->replace_node(conv_weight_node, new_conv_weight_gnode, false);

            // BiasAdd: broadcast -> add
            auto conv_output_shape = conv_node->get_output_shape(0);
            bool is_nhwc = (conv_node_op_ptr->get_data_format() == "NHWC");
            nnfusion::AxisSet broadcast_axes;
            if (is_nhwc)
            {
                for (size_t i = 0; i < conv_output_shape.size() - 1; i++)
                {
                    broadcast_axes.insert(i);
                }
            }
            else
            {
                for (size_t i = 0; i < conv_output_shape.size(); i++)
                {
                    if (i != 1)
                    {
                        broadcast_axes.insert(i);
                    }
                }
            }
            auto new_broadcast_gnode = m_graph->add_node_and_edge(
                std::make_shared<op::Broadcast>(conv_output_shape, broadcast_axes),
                {bn_node->get_in_edge(1)->get_src()});
            shared_ptr<KernelContext> ke_ctx(new KernelContext(new_broadcast_gnode));
            KernelEmitter::Pointer any_op_ke =
                std::make_shared<nnfusion::kernels::cuda::AnyOP>(ke_ctx);
            any_op_ke->get_or_emit_source();
            (*new_broadcast_gnode)["Kernel_Selection_Result"] =
                std::make_pair(NNFusion_DeviceType::CUDA_GPU, any_op_ke);
            ;
            m_nodes.resize(m_graph->get_max_node_id());
            m_nodes[new_broadcast_gnode->get_id()] = std::make_shared<TaggedNode>();
            m_nodes[new_broadcast_gnode->get_id()]->node = new_broadcast_gnode;
            auto new_add_gnode = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                            {conv_node, new_broadcast_gnode});
            m_nodes.resize(m_graph->get_max_node_id());
            m_nodes[new_add_gnode->get_id()] = std::make_shared<TaggedNode>();
            m_nodes[new_add_gnode->get_id()]->node = new_add_gnode;

            // delete batchnorm_inference node
            auto bn_out_edges = bn_node->get_out_edges();
            int next_output_id = 0;
            for (auto bn_out_edge : bn_out_edges)
            {
                auto output_id =
                    bn_out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id++;
                if (output_id != Graph::kControlSlot && output_id == 0)
                {
                    new_add_gnode->set_output(
                        output_id, bn_node->get_outputs().at(bn_out_edge->get_src_output()));
                }
                m_graph->add_edge(
                    new_add_gnode, 0, bn_out_edge->get_dst(), bn_out_edge->get_dst_input());
            }
            m_graph->remove_node(bn_node);

            return new_add_gnode->get_id();
        }
        else
        {
            NNFUSION_LOG(NNFUSION_WARNING)
                << "BatchNormInference folding not implemented: " << identifier << ", fallback";
            return -1;
        }
        return -1;
    }

    template <typename T>
    std::vector<T> ConvertConvolutionWeight(const std::vector<double>& conv_weight,
                                            const std::vector<double>& bn_gain,
                                            const std::vector<double>& bn_variance,
                                            const double bn_epsilon)
    {
        std::vector<T> conv_weight_output;
        conv_weight_output.resize(conv_weight.size());

        auto num_out_channel = bn_gain.size();
        auto filter_size = conv_weight.size() / bn_gain.size();
#pragma omp parallel for
        for (auto i = 0; i < num_out_channel; i++)
        {
            for (auto j = 0; j < filter_size; j++)
            {
                conv_weight_output[i * filter_size + j] =
                    (T)(bn_gain[i] * conv_weight[i * filter_size + j] /
                        sqrt((double)(bn_variance[i]) + bn_epsilon));
            }
        }

        return conv_weight_output;
    }

    template <typename T>
    std::vector<T> ConvertConvolutionBias(const std::vector<double>& conv_bias,
                                          const std::vector<double>& bn_gain,
                                          const std::vector<double>& bn_bias,
                                          const std::vector<double>& bn_mean,
                                          const std::vector<double>& bn_variance,
                                          const double bn_epsilon)
    {
        std::vector<T> conv_bias_output;
        conv_bias_output.resize(conv_bias.size());

        auto num_element = conv_bias.size();
#pragma omp parallel for
        for (auto i = 0; i < num_element; i++)
        {
            conv_bias_output[i] = (T)(bn_gain[i] * (conv_bias[i] - bn_mean[i]) /
                                          sqrt((double)(bn_variance[i]) + bn_epsilon) +
                                      bn_bias[i]);
        }

        return conv_bias_output;
    }

private:
    std::shared_ptr<Graph> m_graph;
    std::vector<std::string> m_pattern;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
};

bool BatchNormInferenceFoldingPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool folding_flag = FLAGS_fbatchnorm_inference_folding;
    if (folding_flag)
    {
        NNFUSION_LOG(INFO) << "batchnorm inference folding Pass starts up for Graph: "
                           << graph->get_name();
        for (auto pattern : BN_FOLDING_PATTERNS)
        {
            BatchNormInferenceOptimizer optimizer(graph, pattern);
            optimizer.MatchAndFolding();
        }
        // if (FLAGS_fconst_folding_backend != "")
        // {
        //     auto const_folding_optimizer = RuntimeConstantFoldingPass();
        //     const_folding_optimizer.run_on_graph(graph);
        // }
        NNFUSION_LOG(INFO) << "batchnorm inference folding Pass ends for Graph: "
                           << graph->get_name();
    }
    return true;
}
