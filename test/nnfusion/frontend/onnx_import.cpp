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
#include "nnfusion/frontend/onnx_import/onnx.hpp"

using namespace nnfusion;
using Inputs = vector<vector<float>>;
using Outputs = vector<vector<float>>;
using RawInputs = vector<vector<char>>;
using RawOutputs = vector<vector<char>>;

template <typename T, typename S = T>
std::vector<T> convert_from_raw(std::vector<char> src)
{
    NNFUSION_CHECK(src.size() % sizeof(S) == 0);
    S* raw_data_ptr = (S*)src.data();
    auto src_data_size = src.size() / sizeof(S);
    return vector<T>(raw_data_ptr, raw_data_ptr + src_data_size);
}

template <typename T>
std::vector<char> convert_to_raw(std::vector<T> src)
{
    auto raw_size = src.size() * sizeof(T);
    char* src_data_ptr = (char*)src.data();
    return vector<char>(src_data_ptr, src_data_ptr + raw_size);
}

template <typename T, typename T1 = T>
vector<vector<T1>> execute(const shared_ptr<nnfusion::graph::Graph>& graph,
                           vector<vector<T>> args,
                           const string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
    auto res = graph_evaluate->eval<T, T1>(args);

    auto output_gnodes = graph->get_outputs();
    NNFUSION_CHECK(output_gnodes.size() == res.size())
        << "number of outputs and results don't match";

    vector<vector<T1>> result_vectors;
    for (auto output_gnode : output_gnodes)
    {
        auto gonde_res = res[output_gnode->get_unique_name()];
        NNFUSION_CHECK(gonde_res.size() == 1);
        result_vectors.push_back((gonde_res[0]));
    }

    return result_vectors;
}

///\todo seems unnecessary, can reuse execute
vector<vector<char>> mixed_type_execute(const shared_ptr<nnfusion::graph::Graph>& graph,
                                        vector<vector<char>> args,
                                        const string& backend_id)
{
    auto parms_gnodes = graph->get_parameters();

    NNFUSION_CHECK(parms_gnodes.size() == args.size())
        << "number of parameters and arguments don't match";

    auto graph_evaluate = make_shared<nnfusion::profiler::GraphEvaluate>(graph, CUDA_GPU);
    auto res = graph_evaluate->mixed_type_eval(args);

    auto output_gnodes = graph->get_outputs();
    vector<vector<char>> result_vectors;
    ///\todo: we don't have output index yet, so we think it's legal either:
    //1. output_gnode size equal to profiler output size
    //2. only have one profiler output, which contains multiple vector
    if (output_gnodes.size() == res.size())
    {
        for (auto output_gnode : output_gnodes)
        {
            auto gonde_res = res[output_gnode->get_unique_name()];
            // remove this constrain
            NNFUSION_CHECK(gonde_res.size() == 1);
            result_vectors.push_back((gonde_res[0]));
        }
    }
    else if (res.size() == 1)
    {
        result_vectors = res.begin()->second;
        NNFUSION_CHECK(result_vectors.size() == output_gnodes.size());
    }
    else
    {
        NNFUSION_CHECK_FAIL() << "input/output count mismatch";
    }

    return result_vectors;
}

TEST(nnfusion_onnx_import, abs_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/abs.onnx"));

    Inputs inputs{{1, -3.4, 0}};
    Outputs expected_outputs{{1, 3.4, 0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, adam_optimizer)
{
    // copy from onnxruntime
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/adam_optimizer_op.onnx"));

    RawInputs raw_inputs;
    // lr
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0.5f}));
    // count
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{3}));
    // w
    raw_inputs.emplace_back(convert_to_raw(vector<float>{1.0f, 2.0f, 3.0f}));
    // g
    raw_inputs.emplace_back(convert_to_raw(vector<float>{4.0f, 5.0f, 6.0f}));
    // m1
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0.1f, 0.2f, 0.3f}));
    // m2
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0.4f, 0.5f, 0.6f}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<int32_t> count_new{convert_from_raw<int32_t>(raw_outputs[0])};
    vector<float> m1_new{convert_from_raw<float>(raw_outputs[1])};
    vector<float> m2_new{convert_from_raw<float>(raw_outputs[2])};
    vector<float> w_new{convert_from_raw<float>(raw_outputs[3])};
    // vector<float> g_new{convert_from_raw<float>(raw_outputs[4])};

    EXPECT_TRUE(count_new == vector<int32_t>{4});
    EXPECT_TRUE(test::all_close_f(m1_new, vector<float>{0.49f, 0.68f, 0.87f}));
    EXPECT_TRUE(test::all_close_f(m2_new, vector<float>{0.4156f, 0.5245f, 0.6354f}));
    EXPECT_TRUE(test::all_close_f(w_new, vector<float>{0.6199609f, 1.5305318f, 2.4542853f}));
    // EXPECT_TRUE(test::all_close_f(g_new, vector<float>{-0.3800391f, -0.4694682f, -0.5457147f}));
}

TEST(nnfusion_onnx_import, add_abc_initializers_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_initializers.onnx"));
    // A + B + C, A is in initializer {1,2,3,4} B is constant {1,2,3,4}
    Inputs inputs{{1, 2, 3, 4}};
    Outputs expected_outputs{{3, 6, 9, 12}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, add_abc_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx"));

    Inputs inputs{{1}, {2}, {3}};
    Outputs expected_outputs{{6}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, add_bcast_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/add_bcast.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>(
                            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
                             {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}})
                            .get_vector());

    inputs.emplace_back(test::NDArray<float, 1>({1, 2, 3, 4, 5}).get_vector());

    Outputs expected_outputs{
        test::NDArray<float, 4>(
            {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
              {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}}})
            .get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, addmul_abc_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/addmul_abc.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{9, 10}, {11, 12}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{5, 6}, {7, 8}}}).get_vector());
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2}, {3, 4}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{46, 62}, {80, 100}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, acos_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/acos.onnx"));

    Inputs inputs{{0, 1, -0.5}};
    Outputs expected_outputs{{1.5708, 0.0, 2.0944}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, and_op)
{
    // cast op is used
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/and.onnx"));

    vector<vector<int64_t>> inputs{{1, 0, 1, 0}, {1, 1, 0, 0}};
    vector<vector<int64_t>> expected_outputs{{1, 0, 0, 0}};

    vector<vector<int64_t>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, asin_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/asin.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.5236, 0.0000, 1.5708}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
/* no kernel implemented for argmax and argmin
TEST(nnfusion_onnx_import, argmax_int32_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmax_int32.onnx"));

    vector<vector<int32_t>> inputs{
        vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    vector<vector<int64_t>> expected_outputs{
        vector<int64_t>{1, 1, 1, 1, 1, 1}};

    vector<vector<int64_t>> outputs{execute<int32_t, int64_t>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, argmin_int32_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_int32.onnx"));

    vector<vector<int32_t>> inputs{
        vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};

    vector<vector<int64_t>> expected_outputs{vector<int64_t>{0, 0, 0, 0}};

    vector<vector<int64_t>> outputs{execute<int32_t, int64_t>(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, argmin_no_keepdims)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/argmin_no_keepdims.onnx"));

    Inputs inputs{{2, 1, 3, 10}};
    Outputs expected_outputs{{1, 0}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}
*/
TEST(nnfusion_onnx_import, atan_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/atan.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.4636, 0.0000, 0.7854}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, average_pool_2d_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 2, 2)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{2.5f, 4.5f}, {10.5f, 12.5f}}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, average_pool_2d_pads_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/average_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{0.f, 1.5f, 3.f}, {6.f, 7.5f, 9.f}, {12.f, 13.5f, 15.f}}}})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, batch_norm_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/batchnorm_default.onnx"));

    Inputs inputs;
    inputs.push_back({-1.f, 0.f, 1.f, 2.f, 3.f, 4.f}); // data {1, 2, 1, 3}
    inputs.push_back({1.f, 1.5f});                     // scale
    inputs.push_back({0.f, 1.f});                      // bias
    inputs.push_back({0.f, 3.f});                      // mean
    inputs.push_back({1.f, 1.5f});                     // var

    Outputs expected_outputs{{-0.999995f, 0.f, 0.999995f, -0.22474074f, 1.f, 2.2247407f}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, ceil_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/ceil.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0., 0., 1.}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, concat_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/concat.onnx"));

    Inputs inputs;

    inputs.emplace_back(test::NDArray<float, 1>({1, 2}).get_vector());
    inputs.emplace_back(test::NDArray<float, 1>({3, 4}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 1>({1, 2, 3, 4}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, cos_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/cos.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.8776, 1.0000, 0.5403}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, custom_op)
{
    // model ctx: input -> one_hot of depth 10 -> add_one
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/custom_op.onnx"));
    size_t depth = 10;

    RawInputs raw_inputs;
    raw_inputs.emplace_back(convert_to_raw(vector<int64_t>{3, 1, 8}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs[0])};

    vector<float> expect_out(3 * depth, 1);
    expect_out[0 * depth + 3] += 1;
    expect_out[1 * depth + 1] += 1;
    expect_out[2 * depth + 8] += 1;

    EXPECT_TRUE(test::all_close_f(out, expect_out));
}

TEST(nnfusion_onnx_import, div_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{1, 4, 12}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{1, 0.5, 0.25}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, exp_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/exp.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.6065, 1.0000, 2.7183}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, floor_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/floor.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-1, 0, 1}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gather_op_axis_0)
{
    // gather along axis 0
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_axis_0.onnx"));

    RawInputs raw_inputs;
    // (3, 3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}));
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1LL, 0LL, 2LL, 1LL}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (2, 2, 3)
    Outputs expected_outputs{
        {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f, 2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gather_op_axis_1)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_axis_1.onnx"));

    RawInputs raw_inputs;
    // (3, 3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}));
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1LL, 0LL, 2LL, 1LL}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (3, 2, 2)
    Outputs expected_outputs{
        {0.1f, 0.0f, 0.2f, 0.1f, 1.1f, 1.0f, 1.2f, 1.1f, 2.1f, 2.0f, 2.2f, 2.1f}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gather_grad_op_axis_0)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gather_grad_axis_0.onnx"));

    RawInputs raw_inputs;
    // // shape, (2), (3, 3)
    // raw_inputs.emplace_back(
    //     convert_to_raw(vector<int32_t>{3, 3}));
    // indices, (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0LL, 1LL, 0LL, 1LL}));
    // grad, (2, 2, 3)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (3, 3)
    Outputs expected_outputs{{0, 2, 4, 6, 8, 10, 0, 0, 0}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gather_nd_op_1)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_nd_1.onnx"));

    RawInputs raw_inputs;
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 2, 3}));
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 0, 1, 1}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto outputs{convert_from_raw<int32_t>(raw_outputs[0])};
    // (2)
    vector<int> expected_outputs{0, 3};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, gather_nd_op_2)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_nd_2.onnx"));

    RawInputs raw_inputs;
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 2, 3}));
    // (2, 1)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1, 0}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto outputs{convert_from_raw<int32_t>(raw_outputs[0])};
    // (2, 2)
    vector<int> expected_outputs{2, 3, 0, 1};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, gather_nd_op_3)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_nd_3.onnx"));

    RawInputs raw_inputs;
    // (2, 2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    // (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 1, 0}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto outputs{convert_from_raw<int32_t>(raw_outputs[0])};
    // (2, 2)
    vector<int> expected_outputs{2, 3, 4, 5};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, gather_nd_op_4)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_nd_4.onnx"));

    RawInputs raw_inputs;
    // (2, 2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    // (2, 1, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 1, 0}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto outputs{convert_from_raw<int32_t>(raw_outputs[0])};
    // (2, 1, 2)
    vector<int> expected_outputs{2, 3, 4, 5};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, gather_nd_op_5)
{
    // batch_dims = 1
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_nd_5.onnx"));

    RawInputs raw_inputs;
    // (2, 2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    // (2, 1)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1, 0}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto outputs{convert_from_raw<int32_t>(raw_outputs[0])};
    // (2, 2)
    vector<int> expected_outputs{2, 3, 4, 5};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, gathernd_grad_op_axis_0)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gathernd_grad_axis_0.onnx"));

    RawInputs raw_inputs;
    // // shape, (3) (2, 2, 3)
    // raw_inputs.emplace_back(
    //     convert_to_raw(vector<int32_t>{3, 3}));
    // indices, (2, 2)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{0LL, 1LL, 1LL, 0LL}));
    // grad, (2, 3)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{1, 2, 3, 4, 5, 6}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (2, 2, 3)
    Outputs expected_outputs{{0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gathernd_grad_op_axis_1)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gathernd_grad_axis_1.onnx"));

    RawInputs raw_inputs;
    // // shape, (3) (2, 2, 3)
    // raw_inputs.emplace_back(
    //     convert_to_raw(vector<int32_t>{3, 3}));
    // indices, (2, 1, 1)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1LL, 0LL}));
    // grad, (2, 3)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{1, 2, 3, 4, 5, 6}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (2, 2, 3)
    Outputs expected_outputs{{0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gathernd_grad_op_axis_2)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gathernd_grad_axis_2.onnx"));

    RawInputs raw_inputs;
    // // shape, (4) (2, 2, 2, 3)
    // raw_inputs.emplace_back(
    //     convert_to_raw(vector<int32_t>{3, 3}));
    // indices, (2, 2, 1)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{1LL, 1LL, 0LL, 1LL}));
    // grad, (2, 2, 3)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (2, 2, 2, 3)
    Outputs expected_outputs{{
        0, 0, 0, 0, 1,  2,  // batch 0
        0, 0, 0, 3, 4,  5,  // batch 1
        6, 7, 8, 0, 0,  0,  // batch 2
        0, 0, 0, 9, 10, 11, // batch 3
    }};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, gemm_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gemm_op.onnx"));

    RawInputs raw_inputs;
    // a, (4, 3)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{0.736117,
                                                         0.716200,
                                                         0.373398,
                                                         0.261163,
                                                         0.220859,
                                                         0.307344,
                                                         0.728187,
                                                         0.569954,
                                                         0.715242,
                                                         0.523189,
                                                         0.679213,
                                                         0.881296}));
    // b, (5, 4)
    raw_inputs.emplace_back(convert_to_raw(
        vector<float>{0.281591, 0.897486, 0.969975, 0.639050, 0.717340, 0.045627, 0.422450,
                      0.362448, 0.515828, 0.087162, 0.719849, 0.457718, 0.303687, 0.332840,
                      0.022761, 0.913949, 0.101926, 0.779844, 0.189475, 0.581422}));
    // c, (1, 5)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{0.370893, 0.820984, 0.792165, 0.393056, 0.793784}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    Outputs outputs{convert_from_raw<float>(raw_outputs[0])};
    // (3, 5)
    Outputs expected_outputs{{0.500398,
                              0.546648,
                              0.568790,
                              0.338874,
                              0.458040,
                              0.476509,
                              0.540042,
                              0.554721,
                              0.368757,
                              0.464859,
                              0.539298,
                              0.513208,
                              0.561670,
                              0.396927,
                              0.509241}};

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, layer_norm_op)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/layer_norm_op.onnx"));

    RawInputs raw_inputs;
    // x (2, 3, 4)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{
        -3.746516, -1.684988, -9.202435, 6.570548,  -9.054055, -1.860990, 9.399231, 3.142350,
        -7.914620, -7.879809, 4.793375,  -2.350031, 2.730476,  -4.923367, 3.219690, -0.335356,
        6.149919,  4.407688,  6.558051,  -1.784896, -8.713762, -3.415960, 0.356440, -6.317601}));
    // scale (4)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{-2.628836, 6.568611, 8.320792, 1.056384}));
    // bias (4)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{-9.189506, 6.937095, -3.509123, 2.068400}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // out (2, 3, 4)
    vector<float> out{convert_from_raw<float>(raw_outputs[0])};
    // mean (2, 3, 1)
    vector<float> mean{convert_from_raw<float>(raw_outputs[1])};
    // inv_std_var (2, 3, 1)
    vector<float> inv_var{convert_from_raw<float>(raw_outputs[2])};

    EXPECT_TRUE(test::all_close_f(
        out, vector<float>{-8.386733,  7.320567,  -14.060352, 3.668873, -5.512508,  4.734921,
                           7.553470,   2.495668,  -6.881125,  1.213067, 9.471450,   2.268590,
                           -11.263096, -3.386870, 4.309620,   1.902825, -11.012063, 8.067120,
                           3.275682,   0.292901,  -5.934144,  9.085129, 8.486524,   1.508164}));
    EXPECT_TRUE(test::all_close_f(
        mean, vector<float>{-2.015848, 0.406634, -3.337771, 0.172861, 3.832691, -4.522721}));
    EXPECT_TRUE(test::all_close_f(
        inv_var, vector<float>{0.176448, 0.147845, 0.191857, 0.308407, 0.299191, 0.295470}));
}

TEST(nnfusion_onnx_import, layer_norm_grad_op)
{
    // copy from onnxruntime
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/layer_norm_grad_op.onnx"));

    RawInputs raw_inputs;
    // dout (2, 3, 4)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{
        -9.889713, 4.293637,  -0.531595, -1.867394, 0.312331,  7.507808, 5.684879, 9.794285,
        9.963729,  -8.451398, -1.821811, -9.688374, -6.054464, 5.743388, 1.490581, -7.043674,
        1.145193,  1.614964,  3.044850,  -8.819752, 5.158533,  4.752753, 3.101693, -7.974546}));
    // x (2, 3, 4)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{
        -3.746516, -1.684988, -9.202435, 6.570548,  -9.054055, -1.860990, 9.399231, 3.142350,
        -7.914620, -7.879809, 4.793375,  -2.350031, 2.730476,  -4.923367, 3.219690, -0.335356,
        6.149919,  4.407688,  6.558051,  -1.784896, -8.713762, -3.415960, 0.356440, -6.317601}));
    // scale (4)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{-2.628836, 6.568611, 8.320792, 1.056384}));
    // mean (2, 3, 1)
    raw_inputs.emplace_back(convert_to_raw(
        vector<float>{-2.015848, 0.406634, -3.337771, 0.172861, 3.832691, -4.522721}));
    // inv_var (2, 3, 1)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{0.176448, 0.147845, 0.191857, 0.308407, 0.299191, 0.295470}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // dx (2, 3, 4)
    vector<float> dx{convert_from_raw<float>(raw_outputs[0])};
    // dscale (4)
    vector<float> dscale{convert_from_raw<float>(raw_outputs[1])};
    // dbias (4)
    vector<float> dbias{convert_from_raw<float>(raw_outputs[2])};

    EXPECT_TRUE(test::all_close_f(
        dx, vector<float>{2.429091, 2.877050,  -3.094702, -2.211439, -1.372361, 4.008402,
                          0.530222, -3.166262, 2.040173,  -3.599919, -1.197469, 2.757214,
                          2.453039, 3.006092,  1.764068,  -7.223199, -4.536900, 0.943452,
                          3.614681, -0.021233, -0.313362, 4.976852,  -2.276259, -2.387231}));
    EXPECT_TRUE(
        test::all_close_f(dscale, vector<float>{-16.535650, -2.096560, 13.745125, 19.453054}));
    EXPECT_TRUE(
        test::all_close_f(dbias, vector<float>{0.635609, 15.461152, 10.968597, -25.599455}));
}

TEST(nnfusion_onnx_import, log_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/log.onnx"));

    Inputs inputs{{0.5, 1, 2}};
    Outputs expected_outputs{{-0.6931, 0.0000, 0.6931}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, matmul_op_1)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_1.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (3, 1, 1, 2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 6}));
    // input1 (2, 2, 2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 8}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (3, 2, 1, 2)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(
        test::all_close_f(output, vector<float>{2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}));
}

TEST(nnfusion_onnx_import, matmul_op_2)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_2.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (2, 3, 2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));
    // input1 (3, 2, 2, 1)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (3, 2, 3, 1)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(
        output,
        vector<float>{1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}));
}

TEST(nnfusion_onnx_import, matmul_op_3)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_3.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 2}));
    // input1 (3, 2, 1)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 6}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (3, 1)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(output, vector<float>{1, 3, 5}));
}

TEST(nnfusion_onnx_import, matmul_op_4)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_4.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (3, 1, 2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 6}));
    // input1 (2)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 2}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (3, 1)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(output, vector<float>{1, 3, 5}));
}

TEST(nnfusion_onnx_import, matmul_op_5)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_5.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 3}));
    // input1 (3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 3}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (,)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(output, vector<float>{5}));
}

TEST(nnfusion_onnx_import, matmul_op_6)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_6.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (3, 4)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));
    // input1 (4, 3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (3, 3)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(output, vector<float>{42, 48, 54, 114, 136, 158, 186, 224, 262}));
}

TEST(nnfusion_onnx_import, matmul_op_7)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_7.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (2, 2, 3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));
    // input1 (3, 4)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (2, 2, 4)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(
        output,
        vector<float>{20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}));
}

TEST(nnfusion_onnx_import, matmul_op_8)
{
    // copy from onnxruntime
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/matmul_op_8.onnx"));

    std::vector<float> common_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    RawInputs raw_inputs;
    // input0 (2, 2, 3)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));
    // input1 (1, 3, 4)
    raw_inputs.emplace_back(
        convert_to_raw(vector<float>{common_input.begin(), common_input.begin() + 12}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (2, 2, 4)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(
        output,
        vector<float>{20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}));
}

TEST(nnfusion_onnx_import, max_pool_2d_pads_op)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/max_pool_2d_pads.onnx"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());

    // (1, 1, 3, 3)
    Outputs expected_outputs{
        test::NDArray<float, 4>({{{{0.f, 2.f, 3.f}, {8.f, 10.f, 11.f}, {12.f, 14.f, 15.f}}}})
            .get_vector()};
    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, pow_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/pow.onnx"));

    vector<vector<int64_t>> inputs{{1, 2, 4}, {3, 1, 2}};
    vector<vector<int64_t>> expected_outputs{{1, 2, 16}};

    vector<vector<int64_t>> outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_EQ(expected_outputs[i], outputs[i]);
    }
}

TEST(nnfusion_onnx_import, relu_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/relu.onnx"));

    Inputs inputs{{-0.5, 0, 1, -1.2, 2.4, -5}};
    Outputs expected_outputs{{0.0000, 0.0000, 1.0000, 0.0000, 2.4000, 0.0000}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sigmoid_op)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{0.3775, 0.5000, 0.7311}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sin_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sin.onnx"));

    Inputs inputs{{-0.5, 0, 1}};
    Outputs expected_outputs{{-0.4794, 0.0000, 0.8415}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sparse_softmax_cross_entropy_op)
{
    // copy from onnxruntime
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sparse_softmax_cross_entropy_op.onnx"));

    RawInputs raw_inputs;
    // x (3, 5)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{-0.9468f,
                                                         1.3250f,
                                                         1.0438f,
                                                         0.4106f,
                                                         -0.2150f,
                                                         -0.3399f,
                                                         -0.4396f,
                                                         1.1835f,
                                                         1.2089f,
                                                         -1.0617f,
                                                         -0.5239f,
                                                         -0.2767f,
                                                         0.9910f,
                                                         -1.5688f,
                                                         -0.2863f}));
    // index (3)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{3, 4, 1}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // output (,)
    vector<float> output{convert_from_raw<float>(raw_outputs[0])};
    // log_prob (3, 5)
    vector<float> log_prob{convert_from_raw<float>(raw_outputs[1])};

    EXPECT_TRUE(test::all_close_f(output, vector<float>{2.2956f}));
    EXPECT_TRUE(test::all_close_f(log_prob,
                                  vector<float>{-3.1773f,
                                                -0.9054f,
                                                -1.1867f,
                                                -1.8199f,
                                                -2.4454f,
                                                -2.4583f,
                                                -2.5580f,
                                                -0.9349f,
                                                -0.9094f,
                                                -3.1800f,
                                                -2.1341f,
                                                -1.8869f,
                                                -0.6192f,
                                                -3.1789f,
                                                -1.8965f}));
}

TEST(nnfusion_onnx_import, sparse_softmax_cross_entropy_grad_op)
{
    // copy from onnxruntime
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sparse_softmax_cross_entropy_grad_op.onnx"));

    RawInputs raw_inputs;
    // dout (,)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{1.5}));
    // log_prob (3, 5)
    raw_inputs.emplace_back(convert_to_raw(vector<float>{-3.1773f,
                                                         -0.9054f,
                                                         -1.1867f,
                                                         -1.8199f,
                                                         -2.4454f,
                                                         -2.4583f,
                                                         -2.5580f,
                                                         -0.9349f,
                                                         -0.9094f,
                                                         -3.1800f,
                                                         -2.1341f,
                                                         -1.8869f,
                                                         -0.6192f,
                                                         -3.1789f,
                                                         -1.8965f}));
    // index (3)
    raw_inputs.emplace_back(convert_to_raw(vector<int32_t>{3, 4, 1}));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    // log_prob (3, 5)
    vector<float> dx{convert_from_raw<float>(raw_outputs[0])};

    EXPECT_TRUE(test::all_close_f(dx,
                                  vector<float>{0.020849,
                                                0.202172,
                                                0.152615,
                                                -0.418978,
                                                0.043342,
                                                0.042791,
                                                0.038731,
                                                0.196318,
                                                0.201368,
                                                -0.479209,
                                                0.059176,
                                                -0.424229,
                                                0.269191,
                                                0.020814,
                                                0.075048}));
}

TEST(nnfusion_onnx_import, sqrt_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sqrt.onnx"));

    Inputs inputs{{0.0, 1.0, 4.0, 5.0}};
    Outputs expected_outputs{{0.0000, 1.0000, 2.0000, 2.2361}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, sub_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sub.onnx"));

    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 3>({{{1, 2, 3}}}).get_vector());

    inputs.emplace_back(test::NDArray<float, 3>({{{4, 5, 7}}}).get_vector());

    Outputs expected_outputs{test::NDArray<float, 3>({{{-3, -3, -4}}}).get_vector()};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, tan_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tan.onnx"));

    Inputs inputs{{-1, 0, 1}};
    Outputs expected_outputs{{-1.5574, 0.0000, 1.5574}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, tanh_op)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.onnx"));

    Inputs inputs{{-1, 0, 1}};
    Outputs expected_outputs{{-0.7616, 0.0000, 0.7616}};

    Outputs outputs{execute(model, inputs, "NNFusion")};
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (size_t i = 0; i < expected_outputs.size(); ++i)
    {
        EXPECT_TRUE(test::all_close_f(expected_outputs[i], outputs[i]));
    }
}

TEST(nnfusion_onnx_import, trainable_dropout_op)
{
    // copy from onnxruntime
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/trainable_dropout.onnx"));

    float dropout_raito = 0.5;
    vector<float> input(2 * 128 * 1024, 1);
    RawInputs raw_inputs;
    // (2, 128, 1024)
    raw_inputs.emplace_back(convert_to_raw(input));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    auto output = convert_from_raw<float>(raw_outputs.at(0));
    vector<char> mask(output.size());
    int n = output.size();
    for (auto i = 0; i < output.size(); i++)
    {
        int byte_index = i / 8;
        int bit_index = i % 8;
        ///\todo: might consider big/little endian
        mask[i] = (raw_outputs[1][byte_index] >> bit_index) & 1;
    }

    EXPECT_EQ(output.size(), input.size());
    EXPECT_EQ(mask.size(), input.size());
    int num_output_zero = 0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (mask[i])
        {
            EXPECT_EQ(output[i], input[i] / (1 - dropout_raito));
        }
        else
        {
            EXPECT_EQ(output[i], 0);
            num_output_zero++;
        }
    }

    auto ratio = (float)num_output_zero / input.size();
    EXPECT_NEAR(ratio, dropout_raito, 0.02);
}