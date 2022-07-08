//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/engine/util/file_util.hpp"
#include "nnfusion/frontend/tensorflow_import/tensorflow.hpp"

using namespace nnfusion;
using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

vector<vector<bool>> char_to_bool(vector<vector<char>> chars)
{
    vector<vector<bool>> ret(chars.size());
    for (size_t i = 0; i < ret.size(); i++)
    {
        for (size_t j = 0; j < chars[i].size(); j++)
        {
            ret[i].push_back(bool(chars[i][j]));
        }
    }
    return ret;
}

template <typename T, typename T1 = T>
std::vector<std::vector<T1>> execute(const std::shared_ptr<nnfusion::graph::Graph>& graph,
                                     std::vector<std::vector<T>> args,
                                     const std::string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    NNFusion_DeviceType dt = CUDA_GPU;
    if (backend_id == "NNFusion::CUDA_GPU")
        dt = CUDA_GPU;
    else if (backend_id == "NNFusion::ROCM_GPU")
        dt = ROCM_GPU;
    else if (backend_id == "NNFusion::GENERIC_CPU")
        dt = GENERIC_CPU;

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, dt);
    auto res = graph_evaluate->eval<T, T1>(args);

    auto output_gnodes = graph->get_outputs();
    NNFUSION_CHECK(output_gnodes.size() == res.size())
        << "number of outputs and results don't match";

    std::vector<std::vector<T1>> result_vectors;
    for (auto output_gnode : output_gnodes)
    {
        auto gonde_res = res[output_gnode->get_unique_name()];
        NNFUSION_CHECK(gonde_res.size() == 1);
        result_vectors.push_back((gonde_res[0]));
    }

    return result_vectors;
}

/* todo: support int64
TEST(nnfusion_tensorflow_import, abs_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_abs_graph.pb"));

    std::vector<std::vector<int64_t>> inputs{};
    std::vector<std::vector<int64_t>> expected_outputs{{2147483649}};

    // constant input is -2147483649
    std::vector<std::vector<int64_t>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {

        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
TEST(nnfusion_tensorflow_import, floormod_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_floor_mod.pb"));

    Inputs inputs;

    inputs.emplace_back(
        test::NDArray<float, 2>{{1.8, 2.2}, {-1.3, -0.04}, {3.0, -12}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 1>{100, -100}.get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 2>{{1.8, -97.8}, {98.7, -0.04}, {3, -12}}.get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, exp_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_exp_graph.pb"));

    Inputs inputs{};
    Outputs expected_outputs{{2.7182817}};

    // constant input is -1.0
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, add_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_add_graph.pb"));

    Inputs inputs{{2}};
    Outputs expected_outputs{{3.0, 4, 5.0}};

    // input add [1.0,2.0,3.0]
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, exp_placeholder_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_exp_placeholder_graph.pb"));

    // the shape of placehoder is (3,2), flatting shape for input and output
    Inputs inputs{test::NDArray<float, 2>{{2, 2}, {2, 3}, {3, 4}}.get_vector()};

    Outputs expected_outputs{test::NDArray<float, 2>{
        {7.389056, 7.389056},
        {7.389056, 20.085537},
        {20.085537, 54.59815}}.get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, cast_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_cast_float32_to_int32_graph.pb"));
    // the shape of placehoder is (3,2), flatting shape for input and output$
    Inputs inputs{test::NDArray<float, 2>{{1.8, 2.2}, {-1.3, -0.04}, {3.0, -12}}.get_vector()};
    std::vector<std::vector<int>> expected_outputs{
        test::NDArray<int, 2>{{1, 2}, {-1, 0}, {3, -12}}.get_vector()};

    std::vector<std::vector<int>> outputs{execute<float, int>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, reshape_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reshape_int64_graph.pb"));
    Inputs inputs{test::NDArray<float, 3>{
        {{1, 1, 1}, {2, 2, 2}}, {{3, 3, 3}, {4, 4, 4}}, {{5, 5, 5}, {6, 6, 6}}}
                      .get_vector()};
    Outputs expected_outputs{test::NDArray<float, 3>{{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
                                                     {{4, 4, 4}, {5, 5, 5}, {6, 6, 6}}}
                                 .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

// TODO: the shape for reshape is placeholder
/*
TEST(nnfusion_tensorflow_import, reshape_placeholder_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reshape_placeholder_graph.pb"));
    Inputs inputs{test::NDArray<float, 3>{{{1, 1, 1}, {2, 2, 2}}, {{3, 3,3},{4,4,4}},{{5,5,5},{6,6,6}}}.get_vector(), {2,-1,3}};
    Outputs expected_outputs{test::NDArray<float, 3>{{{1, 1, 1}, {2, 2, 2}, {3, 3,3}},{{4,4,4},{5,5,5},{6,6,6}}}.get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
TEST(nnfusion_tensorflow_import, relu_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_relu_graph.pb"));
    // the shape of placehoder is (5,1), flatting shape for input and output
    Inputs inputs{{-1, -0.00001, 0, 0.00001, 2}};
    Outputs expected_outputs{{0, 0, 0, 0.00001, 2}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, max_pool_op)
{
    // Pooling with strides=2 and padding=1
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/max_pool_2d_pads.pb"));
    EXPECT_EQ(1, 1);

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());
    // (1, 1, 2, 2)
    auto expected_output = test::NDArray<float, 4>({{{{5.f, 7.f}, {13.f, 15.f}}}}).get_vector();

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_TRUE(test::all_close_f(expected_output, outputs[0]));
}

TEST(nnfusion_tensorflow_import, matmul_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_matmul_graph.pb"));
    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    inputs.emplace_back(
        test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}})
            .get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

/* [WARNING] 2019-12-09T08:33:17z src/nnfusion/core/kernels/cuda_gpu/kernels/convolution.cpp 52    Asymetric padding is not supported by now.
TEST(nnfusion_tensorflow_import, conv2d_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_conv2d_nhwc_graph.pb"));
    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 1>({2.,  -3., -1., -4., -4., 1.,  -5., 1., -4., -5., -5., 1.,
                                 -1., 1.,  3.,  1.,  -1., 4.,  4.,  2., 1.,  -4., 0.,  4.,
                                 1.,  1.,  -4., -4., -4., 0.,  -1., 3., 0.,  -4., -1., 1.,
                                 -3., -1., -5., -1., 3.,  -3., -3., 0., 4.,  -5., 1.,  0.})
            .get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 1>({-35., -10., 11.,  9.,   58.,  33.,  -10., 12., -42., -8.,
                                 36.,  14.,  1.,   -21., -25., -37., -20., 44., -31., -58.,
                                 -36., -9.,  -1.,  7.,   24.,  -34., -23., 4.,  17.,  -32.,
                                 -34., -27., -30., 5.,   -1.,  -15., 37.,  39., -30., 7.,
                                 -31., -32., 4.,   4.,   -2.,  15.,  15.,  -14.})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/

TEST(nnfusion_tensorflow_import, depthwise_conv2d_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_depthwise_conv2d_graph.pb"));
    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 4>{{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}}}
            .get_vector());
    inputs.emplace_back(test::NDArray<float, 4>{{{{1}, {2}}, {{7}, {8}}, {{13}, {14}}},
                                                {{{3}, {4}}, {{9}, {10}}, {{15}, {16}}},
                                                {{{5}, {6}}, {{11}, {12}}, {{17}, {18}}}}
                            .get_vector());
    Outputs expected_outputs{test::NDArray<float, 3>{
        {{228, 300}, {132, 180}},
        {{482, 596}, {266, 344}},
        {{372, 452}, {180, 236}}}.get_vector()};

    Outputs outputs{execute<float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_tensorflow_import, bias_add_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_bias_add_graph.pb"));
    // the shape of placehoder is (3,2), flatting shape for input and output$
    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>{{1.8, 2.2}, {-1.3, -0.04}, {3.0, -12}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 1>{100, -100}.get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 2>{{101.8, -97.8}, {98.7, -100.04}, {103, -112}}.get_vector()};

    Outputs outputs{execute<float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_tensorflow_import, bias_add_grad_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_bias_add_grad_graph.pb"));

    // input data is [1024,50] * 1
    Inputs inputs{};

    auto expected_outputs =
        test::NDArray<float, 1>({1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024})
            .get_vector();
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_TRUE(test::all_close_f(expected_outputs, outputs[0]));
}

TEST(nnfusion_tensorflow_import, avg_pool_op)
{
    // Avg pool with strides=[1,2,2,1], ksize=[1,2,2,1], padding="SAME"
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_avg_pool_graph.pb"));

    // input data shape (1, 4, 4, 1)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{1}, {2}, {3}, {4}},
                                               {{5}, {6}, {7}, {8}},
                                               {{9}, {10}, {11}, {12}},
                                               {{13}, {14}, {15}, {16}}}})
                         .get_vector());
    // (1, 2, 2, 1)
    auto expected_outputs =
        test::NDArray<float, 4>({{{{3.5f}, {5.5f}}, {{11.5f}, {13.5f}}}}).get_vector();

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_TRUE(test::all_close_f(expected_outputs, outputs[0]));
}

TEST(nnfusion_tensorflow_import, fill_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_fill_graph.pb"));

    Inputs inputs{};
    Outputs expected_outputs{
        test::NDArray<float, 2>({{3.5f, 3.5f, 3.5f}, {3.5f, 3.5f, 3.5f}}).get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, pad_op)
{
    // op name "Pad", the padding value is always zero.
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_pad_graph.pb"));

    // paddings is [[1, 1], [2, 2]]
    Inputs inputs{test::NDArray<float, 2>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}).get_vector()};
    Outputs expected_outputs{test::NDArray<float, 2>({{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                                                      {0.f, 0.f, 1.f, 2.f, 3.f, 0.f, 0.f},
                                                      {0.f, 0.f, 4.f, 5.f, 6.f, 0.f, 0.f},
                                                      {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}})
                                 .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, padv2_op)
{
    // if constant values specified in mode "CONSTANT", the op name is "PadV2"
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_padv2_graph.pb"));

    // constant values is 3.0f, paddings is [[1, 1], [2, 2]]
    Inputs inputs{test::NDArray<float, 2>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}).get_vector()};
    Outputs expected_outputs{test::NDArray<float, 2>({{3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f},
                                                      {3.f, 3.f, 1.f, 2.f, 3.f, 3.f, 3.f},
                                                      {3.f, 3.f, 4.f, 5.f, 6.f, 3.f, 3.f},
                                                      {3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f}})
                                 .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

/* error : nnfusion_rt/cuda_codegen/nnfusion_rt.cu(38): error: identifier "cublasHandle_t" is undefined
TEST(nnfusion_tensorflow_import, fused_bn_inference_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_fusedbn_inference_graph.pb"));

    Inputs inputs;
    Outputs expected_outputs{
        test::NDArray<float, 4>(
            {{{{0.35069546}, {0.42758602}, {0.24500477}, {0.32162043}, {0.28218}, {0.26838464}}}})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
*/

TEST(nnfusion_tensorflow_import, concatv2_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_concatv2_graph.pb"));

    // the shape of placehoder is (2,3)
    std::vector<std::vector<int>> inputs{
        test::NDArray<int, 2>{{1, 2, 3}, {4, 5, 6}}.get_vector(),
        test::NDArray<int, 2>{{7, 8, 9}, {10, 11, 12}}.get_vector()};
    std::vector<std::vector<int>> expected_outputs{
        test::NDArray<int, 2>{{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}}.get_vector()};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
/*
      Expected: expected_outputs[i]
      Which is: { 256, 65536, 9, 27 }
To be equal to: outputs.front()
      Which is: { 256, 65535, 9, 27 }
TEST(nnfusion_tensorflow_import, pow_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_pow_graph.pb"));

    std::vector<std::vector<int>> inputs{};
    std::vector<std::vector<int>> expected_outputs{
        test::NDArray<int, 2>{{256, 65536}, {9, 27}}.get_vector()};

    // constant input is [[2,2],[3,3]] and [[8,16],[9,27]]
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
TEST(nnfusion_tensorflow_import, tanh_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_tanh_graph.pb"));

    Inputs inputs{{1.0}};
    Outputs expected_outputs{{0.7615945}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_tensorflow_import, reverse_sequence)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reverse_sequence_graph.pb"));

    vector<vector<int32_t>> inputs;
    vector<vector<int32_t>> expected_outputs{{0, 0, 5, 4, 3, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0,
                                              3, 2, 1, 4, 0, 0, 0, 0, 5, 4, 3, 2, 1, 6, 7, 8}};

    vector<vector<int32_t>> outputs{execute(model, inputs, "NNFusion::CUDA_GPU")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, sigmoid_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_sigmoid_graph.pb"));

    Inputs inputs{{0.5, 1.0}};
    Outputs expected_outputs{{0.62245935, 0.7310586}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, scatteradd_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_scatteradd_graph.pb"));

    Inputs inputs{test::NDArray<float, 2>{{1, 1}, {1, 1}, {-1, -4}, {0, 1}}.get_vector()};
    Outputs expected_outputs{{0, 0, 2, 3, 0, -1, 1, 1}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, scattersub_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_scattersub_graph.pb"));

    Inputs inputs{test::NDArray<float, 2>{{1, 1}, {1, 1}, {-1, -4}, {0, 1}}.get_vector()};
    Outputs expected_outputs{{2, 2, 0, -1, -2, -7, -1, 1}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, scattermax_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_scattermax_graph.pb"));

    Inputs inputs{test::NDArray<float, 2>{{1, 1}, {1, 1}, {-1, -4}, {0, 1}}.get_vector()};
    Outputs expected_outputs{{1, 1, 1, 2, 1, 3, 1, 1}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, scattermin_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_scattermin_graph.pb"));

    Inputs inputs{test::NDArray<float, 2>{{1, 1}, {1, 1}, {-1, -4}, {0, 1}}.get_vector()};
    Outputs expected_outputs{{-1, -1, 1, 1, -1, -4, 0, 0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, apply_momentum_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_apply_momentum_graph.pb"));

    Inputs inputs{};
    Outputs expected_outputs{{0.99596, 0.99798, 0.99798, 0.9899, 0.99394, 0.99596}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

// test unique, shape, stridedslice, unsortedsegmentsum and sparseapplymomentum ops
TEST(nnfusion_tensorflow_import, sparse_apply_momentum_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_sparse_apply_momentum_graph.pb"));

    Inputs inputs{};
    Outputs expected_outputs{{0.9878794, 0.9878794, 0.9959598, 0.9959598, 1.0, 1.0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_tensorflow_import, reduce_sum_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_sum_graph.pb"));

    std::vector<std::vector<int>> inputs{test::NDArray<int, 2>{{1, 1, 1}, {2, 2, 2}}.get_vector()};
    std::vector<std::vector<int>> expected_outputs{test::NDArray<int, 2>{{3}, {6}}.get_vector()};

    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, split_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_split_graph.pb"));

    // input size : {5, 30}, num_or_size_splits : 10, axis : 1,
    std::vector<std::vector<int>> inputs{};
    std::vector<std::vector<int>> expected_outputs{
        {15}, {15}, {15}, {15}, {15}, {15}, {15}, {15}, {15}, {15}};

    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());

    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i][0], outputs[i].size());
    }
}

TEST(nnfusion_tensorflow_import, splitV_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_splitv_graph.pb"));

    // input size : {5, 30}, splits : {3, 9, 18}, axis : 1,
    std::vector<std::vector<int>> inputs{};
    std::vector<std::vector<int>> expected_outputs{{15}, {45}, {90}};

    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i][0], outputs[i].size());
    }
}

TEST(nnfusion_tensorflow_import, split_add_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_split_add_graph.pb"));

    std::vector<std::vector<int>> inputs{};
    std::vector<std::vector<int>> expected_outputs{};
    expected_outputs.emplace_back(std::vector<int>(50, 3));

    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, mean_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_mean_graph.pb"));

    Inputs inputs{test::NDArray<float, 2>{{1, 1, 1, 1}, {2, 2, 2, 2}}.get_vector()};
    Outputs expected_outputs{{1.5, 1.5, 1.5, 1.5}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, slice_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_slice_graph.pb"));

    std::vector<std::vector<int>> inputs{};
    std::vector<std::vector<int>> expected_outputs{
        test::NDArray<int, 2>{{3, 3, 3}, {5, 5, 5}}.get_vector()};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
/*
Expected: expected_outputs[i]
      Which is: { 1 }
To be equal to: outputs[i]
      Which is: { 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, 1107296256, ... }
TEST(nnfusion_tensorflow_import, batch_matmul_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_batch_matmul.pb"));

    std::vector<std::vector<int>> inputs{};

    std::vector<std::vector<int>> expected_outputs{{1}};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
/* todo: support int
TEST(nnfusion_tensorflow_import, transpose_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_transpose_4d.pb"));

    std::vector<std::vector<int>> inputs{};

    std::vector<std::vector<int>> expected_outputs{test::NDArray<int, 4>{
        {{{1, 9, 0, 9, 5},
          {0, 1, 1, 4, 8},
          {6, 5, 1, 7, 1},
          {9, 3, 2, 7, 4},
          {7, 2, 3, 3, 4},
          {7, 4, 0, 3, 0},
          {0, 5, 2, 1, 5}},

         {{2, 0, 1, 7, 5},
          {0, 3, 8, 3, 4},
          {2, 7, 2, 8, 7},
          {3, 1, 6, 0, 8},
          {6, 0, 7, 6, 3},
          {5, 7, 7, 0, 1},
          {2, 8, 1, 3, 8}},

         {{9, 7, 7, 4, 4},
          {4, 5, 7, 1, 5},
          {9, 3, 4, 1, 8},
          {9, 4, 1, 3, 9},
          {1, 4, 0, 9, 6},
          {8, 4, 5, 2, 8},
          {6, 6, 6, 3, 7}},

         {{8, 9, 3, 2, 6},
          {3, 9, 6, 1, 7},
          {6, 4, 4, 0, 2},
          {8, 0, 8, 8, 8},
          {9, 7, 0, 0, 3},
          {7, 3, 8, 0, 4},
          {2, 6, 8, 2, 4}}},

        {{{7, 5, 6, 0, 3},
          {3, 9, 9, 4, 6},
          {6, 8, 0, 3, 0},
          {9, 9, 2, 7, 9},
          {8, 3, 7, 0, 0},
          {5, 3, 8, 1, 9},
          {0, 1, 1, 8, 3}},

         {{8, 3, 3, 8, 8},
          {0, 0, 6, 4, 9},
          {4, 4, 5, 4, 1},
          {0, 7, 7, 8, 1},
          {5, 8, 9, 4, 4},
          {2, 6, 4, 1, 3},
          {0, 5, 1, 4, 3}},

         {{7, 6, 9, 7, 5},
          {8, 3, 3, 1, 7},
          {5, 4, 9, 5, 5},
          {4, 9, 4, 3, 6},
          {6, 8, 7, 6, 4},
          {6, 3, 8, 2, 0},
          {6, 2, 2, 6, 5}},

         {{4, 3, 9, 3, 4},
          {1, 4, 9, 5, 9},
          {7, 7, 9, 7, 1},
          {2, 0, 8, 8, 0},
          {8, 3, 6, 1, 7},
          {2, 0, 1, 3, 9},
          {2, 2, 1, 7, 7}}},

        {{{6, 4, 6, 9, 3},
          {0, 8, 2, 7, 9},
          {6, 1, 8, 8, 0},
          {8, 1, 8, 3, 5},
          {4, 2, 2, 3, 8},
          {5, 5, 0, 5, 1},
          {0, 5, 1, 9, 2}},

         {{4, 9, 6, 9, 6},
          {1, 6, 3, 4, 4},
          {6, 9, 2, 6, 3},
          {2, 4, 4, 8, 4},
          {8, 1, 4, 0, 5},
          {2, 5, 9, 8, 8},
          {8, 4, 4, 5, 6}},

         {{9, 7, 6, 7, 3},
          {0, 4, 0, 4, 4},
          {5, 9, 7, 8, 9},
          {8, 7, 2, 4, 4},
          {2, 9, 3, 9, 8},
          {5, 3, 8, 2, 2},
          {6, 2, 5, 1, 5}},

         {{9, 4, 9, 0, 0},
          {6, 8, 0, 0, 7},
          {5, 3, 1, 0, 1},
          {4, 8, 9, 2, 1},
          {4, 3, 9, 6, 6},
          {2, 7, 7, 3, 8},
          {5, 9, 4, 9, 3}}}}.get_vector()};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
/* todo: support int
TEST(nnfusion_tensorflow_import, one_hot_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_one_hot.pb"));

    std::vector<std::vector<int>> inputs{};

    std::vector<std::vector<int>> expected_outputs{{1}};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/

TEST(nnfusion_tensorflow_import, select_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_select_graph.pb"));

    // input data is [[true, false, true], [false, false, true]]
    // [[1,2,3],[4,5,6]]
    // [[7,8,9],[10,11,12]]
    Inputs inputs{};

    auto expected_outputs = test::NDArray<float, 2>({{1, 8, 3}, {10, 11, 6}}).get_vector();

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_TRUE(test::all_close_f(expected_outputs, outputs[0]));
}

TEST(nnfusion_tensorflow_import, reduce_sum_scale_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_sum_2_graph.pb"));

    // input [[1,2,3],[4,5,6]]
    Inputs inputs{};

    Outputs expected_outputs{{21}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, reduce_sum_null_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_sum_null_graph.pb"));

    // input [[1,2,3],[4,5,6]]
    Inputs inputs{};

    Outputs expected_outputs{{1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, reduce_any_0_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_any_0_graph.pb"));
    vector<vector<char>> inputs{};
    vector<vector<bool>> expected_outputs{{true}};
    vector<vector<bool>> outputs = char_to_bool(execute(model, inputs, "NNFusion::CUDA_GPU"));
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, reduce_any_1_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reduce_any_1_graph.pb"));
    vector<vector<char>> inputs{};
    vector<vector<bool>> expected_outputs{{true, false}};
    vector<vector<bool>> outputs = char_to_bool(execute(model, inputs, "NNFusion::CUDA_GPU"));
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, floordiv_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_floordiv_graph.pb"));

    // input [5, 5, 5] [2, 2, 2]
    std::vector<std::vector<int>> inputs{};

    std::vector<std::vector<int>> expected_outputs{{2, 2, 2}};
    std::vector<std::vector<int>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_tensorflow_import, addn)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_addn_graph.pb"));

    Inputs inputs{{1, 2}, {3, 4}, {5, 6}};

    Outputs expected_outputs{{9, 12}};

    ///\todo Change nnfusion::INTERPRETER into nnfusion::reference
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

/*
      Expected: expected_outputs[i]
      Which is: { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4, ... }
To be equal to: outputs[i]
      Which is: { 1.4013e-45, 2.8026e-45, 4.2039e-45, 5.60519e-45, 1.4013e-45, 2.8026e-45, 4.2039e-45, 5.60519e-45, 1.4013e-45, 2.8026e-45, 4.2039e-45, 5.60519e-45, 7.00649e-45, 8.40779e-45, 9.80909e-45, 1.12104e-44, 7.00649e-45, 8.40779e-45, 9.80909e-45, 1.12104e-44, 7.00649e-45, 8.40779e-45, 9.80909e-45, 1.12104e-44, 1.4013e-45, 2.8026e-45, 4.2039e-45, 5.60519e-45, 1.4013e-45, 2.8026e-45, 4.2039e-45, 5.60519e-45, ... }
TEST(nnfusion_tensorflow_import, tile)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_tile_graph.pb"));

    Inputs inputs{};

    Outputs expected_outputs{{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8,
                              5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4,
                              1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8}};

    ///\todo Change nnfusion::INTERPRETER into nnfusion::reference
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
/*
      Expected: expected_outputs[i]
      Which is: { 5, 5, 5, 5, 5, 6, 7, 8 }
To be equal to: outputs[i]
      Which is: { 7.00649e-45, 7.00649e-45, 7.00649e-45, 7.00649e-45, 7.00649e-45, 8.40779e-45, 9.80909e-45, 1.12104e-44 }
TEST(nnfusion_tensorflow_import, unsorted_segment_sum)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_unsorted_segment_sum_graph.pb"));

    Inputs inputs{};

    Outputs expected_outputs{{5, 5, 5, 5, 5, 6, 7, 8}};

    ///\todo Change nnfusion::INTERPRETER into nnfusion::reference
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}
*/
