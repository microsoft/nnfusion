// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "nnfusion/engine/util/file_util.hpp"
#include "nnfusion/frontend/torchscript_import/torchscript.hpp"

using namespace nnfusion;
using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

template <typename T, typename T1 = T>
std::vector<std::vector<T1>> execute(const std::shared_ptr<nnfusion::graph::Graph>& graph,
                                     std::vector<std::vector<T>> args,
                                     const std::string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
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

TEST(nnfusion_torchscript_import, add_op)
{
    auto params_vec =
        nnfusion::frontend::build_torchscript_params_from_string("1,3:float;1,3:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/add_module.pt"), params_vec);

    Inputs inputs{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};

    Outputs expected_outputs{{2.0, 2.0, 2.0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_torchscript_import, dot_op)
{
    auto params_vec =
        nnfusion::frontend::build_torchscript_params_from_string("1,3:float;1,3:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/dot_module.pt"), params_vec);

    Inputs inputs{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};

    Outputs expected_outputs{{1.0, 1.0, 1.0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_torchscript_import, tanh_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/tanh_trace_module.pt"), params_vec);

    Inputs inputs{{1.0, 1.0}};

    Outputs expected_outputs{{0.7616, 0.7616}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, permute_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2,3,5:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/permute_trace_module.pt"), params_vec);

    // permute order : 2, 0, 1
    Inputs inputs{test::NDArray<float, 3>{{{-0.1032, -0.0960, 0.3521, -0.1121, 1.4110},
                                           {0.3053, 1.4371, -0.1136, 1.2640, -0.3912},
                                           {0.0393, -0.0691, -1.5069, -0.9040, 0.4331}},

                                          {{-1.0872, -0.1319, 0.3586, 0.6140, 0.5213},
                                           {0.1988, -1.1642, -0.5577, 0.7577, 0.2615},
                                           {0.0673, 0.5938, -0.1845, -1.2352, 0.9389}}}
                      .get_vector()};

    Outputs expected_outputs{
        test::NDArray<float, 3>{{{-0.1032, 0.3053, 0.0393}, {-1.0872, 0.1988, 0.0673}},

                                {{-0.0960, 1.4371, -0.0691}, {-0.1319, -1.1642, 0.5938}},

                                {{0.3521, -0.1136, -1.5069}, {0.3586, -0.5577, -0.1845}},

                                {{-0.1121, 1.2640, -0.9040}, {0.6140, 0.7577, -1.2352}},

                                {{1.4110, -0.3912, 0.4331}, {0.5213, 0.2615, 0.9389}}}
            .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, expand_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("3,1:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/expand_trace_module.pt"), params_vec);

    // expand(2, -1, 4)
    Inputs inputs{test::NDArray<float, 2>{{1}, {2}, {3}}.get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},

                                                     {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}}}
                                 .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, softmax_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2,3,5:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/softmax_trace_module.pt"), params_vec);

    // input is ones(2,3,5)
    Inputs inputs{test::NDArray<float, 3>{
        {{1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0}},

        {{1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0}}}
                      .get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{{{0.2000, 0.2000, 0.2000, 0.2000, 0.2000},
                                                      {0.2000, 0.2000, 0.2000, 0.2000, 0.2000},
                                                      {0.2000, 0.2000, 0.2000, 0.2000, 0.2000}},

                                                     {{0.2000, 0.2000, 0.2000, 0.2000, 0.2000},
                                                      {0.2000, 0.2000, 0.2000, 0.2000, 0.2000},
                                                      {0.2000, 0.2000, 0.2000, 0.2000, 0.2000}}}
                                 .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, div_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2,3:float;3:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/div_trace_module.pt"), params_vec);

    // input is ones(2,3), div tensor {1, 2, 0.5}
    Inputs inputs{test::NDArray<float, 2>{{1., 1., 1.}, {1., 1., 1.}}.get_vector()};
    inputs.push_back({1, 2, 0.5});

    Outputs expected_outputs{
        test::NDArray<float, 2>{{1.0000, 0.5000, 2.0000}, {1.0000, 0.5000, 2.0000}}.get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, div_scale_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2,3,5:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/div_scale_trace_module.pt"), params_vec);

    // input is ones(2,3, 5), div scale value 0.5
    Inputs inputs{
        test::NDArray<float, 3>{{{1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}},

                                {{1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}}}
            .get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{
        {{2., 2., 2., 2., 2.}, {2., 2., 2., 2., 2.}, {2., 2., 2., 2., 2.}},

        {{2., 2., 2., 2., 2.},
         {2., 2., 2., 2., 2.},
         {2., 2., 2., 2., 2.}}}.get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, embedding_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("2,4:int32_t");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/embedding_trace_module.pt"), params_vec);

    // embedding(10, 3)
    std::vector<std::vector<int>> inputs{
        test::NDArray<int, 2>{{1, 2, 4, 5}, {4, 3, 2, 9}}.get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{{{-0.4192, 1.1735, 0.9895},
                                                      {-2.3239, 1.8840, 1.4442},
                                                      {-1.5284, 0.5130, -1.2059},
                                                      {1.5348, 2.0210, -1.0781}},

                                                     {{-1.5284, 0.5130, -1.2059},
                                                      {0.4165, 1.2190, 0.5365},
                                                      {-2.3239, 1.8840, 1.4442},
                                                      {0.1149, -1.7654, -0.5618}}}
                                 .get_vector()};

    Outputs outputs{execute<int, float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, embedding_padding_idx_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("1,4:int32_t");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/embedding_padding_idx_trace_module.pt"),
        params_vec);

    // embedding(10, 3)
    std::vector<std::vector<int>> inputs{test::NDArray<int, 2>{{0, 2, 0, 5}}.get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{{{0.0000, 0.0000, 0.0000},
                                                      {0.2948, 0.9842, -1.0738},
                                                      {0.0000, 0.0000, 0.0000},
                                                      {1.1493, -0.9232, 1.4948}}}
                                 .get_vector()};

    Outputs outputs{execute<int, float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_torchscript_import, embedding_pretrained_op)
{
    auto params_vec = nnfusion::frontend::build_torchscript_params_from_string("1:int32_t");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/embedding_pretrained_trace_module.pt"),
        params_vec);

    // weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    std::vector<std::vector<int>> inputs{{1}};

    Outputs expected_outputs{{4.0, 5.1, 6.3}};

    Outputs outputs{execute<int, float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

/*
TEST(nnfusion_torchscript_import, layer_norm_op)
{
    auto params_vec = nnfusion::frontend::build_params_from_string("5,3,2:float");
    auto model = frontend::load_torchscript_model(
        file_util::path_join(SERIALIZED_ZOO, "torchscript/layer_norm_trace_module.pt"),
        params_vec);

        Inputs inputs{
        test::NDArray<float, 3>{{{-0.3140,  0.6681},
         {-0.4574,  0.1095},
         {-1.2250, -1.1414}},

        {{ 0.4774, -0.6298},
         { 0.3483, -0.7385},
         { 1.6204,  0.4718}},

        {{ 0.4680, -0.1596},
         {-1.3661,  1.3688},
         {-1.6232, -1.1741}},

        {{ 0.6587,  0.5479},
         {-1.3072,  0.0645},
         {-0.8138,  0.5599}},

        {{-0.4541, -0.7031},
         { 0.4945,  0.6787},
         { 1.9434,  0.5209}}}
            .get_vector()};

    Outputs expected_outputs{test::NDArray<float, 3>{{{ 0.1196,  1.5999},
         {-0.0965,  0.7579},
         {-1.2534, -1.1274}},

        {{ 0.2776, -1.1249},
         { 0.1140, -1.2626},
         { 1.7254,  0.2705}},

        {{ 0.8188,  0.2364},
         {-0.8832,  1.6547},
         {-1.1218, -0.7050}},

        {{ 0.9380,  0.7910},
         {-1.6701,  0.1497},
         {-1.0155,  0.8069}},

        {{-1.0097, -1.2996},
         { 0.0944,  0.3088},
         { 1.7809,  0.1251}}}
                                 .get_vector()};

    Outputs outputs{execute<int, float>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (std::size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
*/