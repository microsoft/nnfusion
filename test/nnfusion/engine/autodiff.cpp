//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../test_util/common.hpp"
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "nnfusion/engine/pass/graph/autodiff_pass.hpp"
#include "nnfusion/engine/util/file_util.hpp"
#include "nnfusion/frontend/onnx_import/onnx.hpp"

using namespace nnfusion;

namespace
{
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

    void build_backward_graph(std::shared_ptr<nnfusion::graph::Graph>& graph,
                              std::shared_ptr<vector<vector<float>>> backward_inputs = nullptr)
    {
        FLAGS_fautodiff = true;
        FLAGS_ftraining_optimizer = "{\"optimizer\": \"SGD\", \"learning_rate\": 0.1}";
        auto ad_pass = nnfusion::pass::graph::AutodiffPass();
        ad_pass.run_on_graph(graph, backward_inputs);
    }
}

TEST(nnfusion_pass_autodiff, multiply)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/mul.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<float>{4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> b_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{4, 10, 18, 28, 40, 54}));
    EXPECT_TRUE(test::all_close_f(a_grad, b));
    EXPECT_TRUE(test::all_close_f(b_grad, a));
}

TEST(nnfusion_pass_autodiff, divide)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/div.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{8.0f, 4.0f, 2.0f};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<float>{2.0f, 1.0f, 2.0f};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> b_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{4, 4, 1}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{0.5, 1, 0.5}));
    EXPECT_TRUE(test::all_close_f(b_grad, vector<float>{-2, -4, -0.5}));
}

TEST(nnfusion_pass_autodiff, tanh)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tanh.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 2.0f, 3.0f};
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{0.76159416, 0.96402758, 0.99505475}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{0.41994236, 0.07064401, 0.00986506}));
}

TEST(nnfusion_pass_autodiff, gather)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gather_axis_0.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<int32_t>{0, 1, 1, 2};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(
        out, vector<float>{1, 1.2, 1.9, 2.3f, 3.4f, 3.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1}));
}

TEST(nnfusion_pass_autodiff, softmax)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/softmax_axis_1.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out,
                                  vector<float>{0.09003057,
                                                0.24472848,
                                                0.66524094,
                                                0.09003057,
                                                0.24472848,
                                                0.66524094,
                                                0.09003057,
                                                0.24472848,
                                                0.66524094}));
    EXPECT_TRUE(test::all_close_f(a_grad,
                                  vector<float>{1.07325e-08,
                                                2.91739e-08,
                                                7.93029e-08,
                                                1.07325e-08,
                                                2.91739e-08,
                                                7.93029e-08,
                                                1.07325e-08,
                                                2.91739e-08,
                                                7.93029e-08}));
}

TEST(nnfusion_pass_autodiff, batchmatmul)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/batch_mat_mul.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                           5.0f,
                           6.0f,
                           1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                           5.0f,
                           6.0f,
                           1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                           5.0f,
                           6.0f};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<float>{4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> b_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{14, 32, 50, 20, 46, 72, 26, 60, 94}));
    EXPECT_TRUE(test::all_close_f(
        a_grad, vector<float>{4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9}));
    EXPECT_TRUE(test::all_close_f(b_grad, vector<float>{9, 12, 9, 12, 9, 12}));
}

TEST(nnfusion_pass_autodiff, log)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/log.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1.0f, 2.0f, 3.0f};
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{0, 0.6931472, 1.0986123}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{1, 0.5, 0.3333333}));
}

TEST(nnfusion_pass_autodiff, sparse_softmax_cross_entropy)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/sparse_softmax_cross_entropy_op.onnx"));

    model->set_outputs({model->get_outputs()[0]});
    build_backward_graph(model);
    RawInputs raw_inputs;
    // a
    auto a = vector<float>{2.9745677e-01,
                           9.1013080e-01,
                           8.0400240e-01,
                           8.9810324e-01,
                           4.7342436e-04,
                           2.9623753e-01,
                           5.0297123e-01,
                           5.4320157e-01,
                           8.5549527e-01,
                           1.5764821e-01,
                           9.6110272e-01,
                           6.3229793e-01,
                           7.5646895e-01,
                           6.2810576e-01,
                           3.9712179e-02};
    raw_inputs.emplace_back(convert_to_raw(a));
    // b
    auto b = vector<int32_t>{0, 4, 0};
    raw_inputs.emplace_back(convert_to_raw(b));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{1.734120966}));
    EXPECT_TRUE(test::all_close_f(a_grad,
                                  vector<float>{-0.28619399,
                                                0.08698932,
                                                0.07822984,
                                                0.08594902,
                                                0.03502714,
                                                0.05437983,
                                                0.06686788,
                                                0.06961337,
                                                0.09513004,
                                                -0.28599101,
                                                -0.24199581,
                                                0.06574282,
                                                0.07443541,
                                                0.06546833,
                                                0.03634925}));
}

TEST(nnfusion_pass_autodiff, gelu)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/gelu.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{
        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
    };
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(
        out,
        vector<float>{0.8413, 1.9545, 2.9960, 0.8413, 1.9545, 2.9960, 0.8413, 1.9545, 2.9960}));
    EXPECT_TRUE(test::all_close_f(
        a_grad,
        vector<float>{1.0833, 1.0852, 1.0119, 1.0833, 1.0852, 1.0119, 1.0833, 1.0852, 1.0119}));
}

TEST(nnfusion_pass_autodiff, conv)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/conv_with_strides_padding.onnx"));

    model->set_outputs({model->get_outputs()[0]});
    build_backward_graph(model);
    RawInputs raw_inputs;
    // x
    auto x = vector<float>(35);
    for (int i = 0; i < 35; ++i)
        x[i] = i;
    raw_inputs.emplace_back(convert_to_raw(x));
    // w
    auto w = vector<float>{2.9745677e-01,
                           9.1013080e-01,
                           8.0400240e-01,
                           8.9810324e-01,
                           4.7342436e-04,
                           2.9623753e-01,
                           5.0297123e-01,
                           5.4320157e-01,
                           8.5549527e-01};
    raw_inputs.emplace_back(convert_to_raw(w));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> x_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> w_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out,
                                  vector<float>{8.145217e+00,
                                                1.545196e+01,
                                                1.160879e+01,
                                                3.447396e+01,
                                                6.100446e+01,
                                                4.162711e+01,
                                                6.856937e+01,
                                                1.120852e+02,
                                                7.315048e+01,
                                                5.285490e+01,
                                                9.245167e+01,
                                                6.437609e+01}));
    EXPECT_TRUE(test::all_close_f(
        x_grad,
        vector<float>{4.734244e-04, 1.194341e+00, 4.734244e-04, 1.194341e+00, 4.734244e-04,
                      1.453332e+00, 2.459926e+00, 1.453332e+00, 2.459926e+00, 1.453332e+00,
                      4.734244e-04, 1.194341e+00, 4.734244e-04, 1.194341e+00, 4.734244e-04,
                      1.453332e+00, 2.459926e+00, 1.453332e+00, 2.459926e+00, 1.453332e+00,
                      4.734244e-04, 1.194341e+00, 4.734244e-04, 1.194341e+00, 4.734244e-04,
                      1.453332e+00, 2.459926e+00, 1.453332e+00, 2.459926e+00, 1.453332e+00,
                      4.734244e-04, 1.194341e+00, 4.734244e-04, 1.194341e+00, 4.734244e-04}));
    EXPECT_TRUE(test::all_close_f(w_grad,
                                  vector<float>{1.020000e+02,
                                                1.530000e+02,
                                                1.020000e+02,
                                                1.360000e+02,
                                                2.040000e+02,
                                                1.360000e+02,
                                                1.020000e+02,
                                                1.530000e+02,
                                                1.020000e+02}));
}

TEST(nnfusion_pass_autodiff, conv1d)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/conv1d.onnx"));

    model->set_outputs({model->get_outputs()[0]});
    build_backward_graph(model);
    RawInputs raw_inputs;
    // x
    auto x = vector<float>{-0.7583, -0.5220, -0.0367, 0.3002,  0.1444, 1.4162,  -0.1217, 0.2265,
                           0.8248,  -0.1658, -0.6053, -1.0609, 2.3762, 1.0725,  0.1565,

                           0.6086,  -0.3693, -0.4803, 0.7205,  0.4418, 0.4734,  -0.4241, 0.1261,
                           -0.9748, 0.2929,  -1.9605, 0.9493,  0.4774, -0.4049, 0.9748};
    raw_inputs.emplace_back(convert_to_raw(x));
    // w
    auto w = vector<float>{-0.2450, 0.2661,  0.3636,  -0.1040, -0.2286, 0.2135,

                           -0.3568, 0.3563,  0.3599,  -0.3011, -0.0962, -0.0637,

                           -0.1210, -0.0831, -0.0211, 0.1642,  -0.3098, -0.1787,

                           0.1211,  0.4018,  0.2846,  -0.1065, 0.0786,  0.2718};
    raw_inputs.emplace_back(convert_to_raw(w));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> x_grad{convert_from_raw<float>(raw_outputs.at(1))};
    vector<float> w_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(
        out, vector<float>{-0.4783, 0.4863,  0.8002,  -0.2286, 0.0703,  -0.1314, -0.6581, 0.7568,
                           0.0118,  -0.3437, 0.1780,  -0.1263, 0.4037,  0.4624,  0.0100,  -0.8175,
                           -0.4531, -0.0625, -0.6201, -0.2216, 0.4256,  0.5710,  0.4736,  -0.0174,
                           -0.3059, 0.6196,  -0.3196, 0.2610,  -0.1432, -0.2246, 0.1992,  0.0774,
                           -0.3517, 0.7469,  -0.5619, -0.1460, 0.3775,  0.3151,  -0.2652, -0.2400,
                           -0.1040, -0.3616, -0.3387, 0.2091,  -0.1675, 0.2986,  0.1893,  0.2135}));
    EXPECT_TRUE(test::all_close_f(
        x_grad, vector<float>{0.3395, 0.3395,  0.3395,  0.3395,  0.3395,  0.6396,  0.6396,  0.6396,
                              0.6396, 0.6396,  -0.3130, -0.3130, -0.3130, -0.3130, -0.3130, 0.3395,
                              0.3395, 0.3395,  0.3395,  0.3395,  0.6396,  0.6396,  0.6396,  0.6396,
                              0.6396, -0.3130, -0.3130, -0.3130, -0.3130, -0.3130}));
    EXPECT_TRUE(test::all_close_f(
        w_grad, vector<float>{0.0488, 0.0488, 1.6734, 1.6734, 1.9752, 1.9752, 0.0488, 0.0488,
                              1.6734, 1.6734, 1.9752, 1.9752, 0.0488, 0.0488, 1.6734, 1.6734,
                              1.9752, 1.9752, 0.0488, 0.0488, 1.6734, 1.6734, 1.9752, 1.9752}));
}

TEST(nnfusion_pass_autodiff, avg_pool_1d_default)
{
    auto model = frontend::load_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/avg_pool_1d_default.onnx"));

    model->set_outputs({model->get_outputs()[0]});
    build_backward_graph(model);
    auto x = vector<float>{-0.1609, -0.6482, 1.7266,  1.2513,  -1.0219, -1.3534, -0.1036, -1.8076,
                           0.5356,  -0.2011, 0.9565,  -0.3530, 0.2873,  -1.8090, 1.4433,  1.6176,
                           1.8445,  -0.2011, -0.3197, -0.6869, -0.4957, -0.8529, -0.4022, 0.6894,
                           -1.0498, -0.2938, -1.4487, 0.5454,  -1.6465, -1.5261};
    RawInputs raw_inputs;
    raw_inputs.emplace_back(convert_to_raw(x));

    auto y_truth =
        vector<float>{-0.4045, 0.5392,  1.4889,  0.1147,  -0.7285, -0.9556, -0.6360, 0.1672,
                      0.3017,  -0.0328, -0.7608, -0.1828, 1.7310,  0.8217,  -0.2604, -0.5033,
                      -0.6743, -0.6275, 0.1436,  -0.1802, -0.8713, -0.4516, -0.5505, -1.5863};

    auto x_grad_truth = vector<float>{
        0.5000, 1.0000, 1.0000, 1.0000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
        0.5000, 1.0000, 1.0000, 1.0000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 0.5000,
        0.5000, 1.0000, 1.0000, 1.0000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 0.5000};

    auto y_grad_truth = vector<float>{1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> x_grad{convert_from_raw<float>(raw_outputs.at(1))};
    //vector<float> y_grad{convert_from_raw<float>(raw_outputs.at(2))};

    EXPECT_TRUE(test::all_close_f(out, y_truth));
    EXPECT_TRUE(test::all_close_f(x_grad, x_grad_truth));
    //EXPECT_TRUE(test::all_close_f(y_grad, y_grad_truth));
}

TEST(nnfusion_pass_autodiff, abs)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/abs.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{1, -3.4, 0};
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{1, 3.4, 0}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{1, -1, 0}));
}

TEST(nnfusion_pass_autodiff, sigmoid)
{
    auto model =
        frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/sigmoid.onnx"));

    build_backward_graph(model);

    RawInputs raw_inputs;
    // a
    auto a = vector<float>{
        -1.0f, 0.0f, 1.0f,
    };
    raw_inputs.emplace_back(convert_to_raw(a));

    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out, vector<float>{0.2689, 0.5000, 0.7311}));
    EXPECT_TRUE(test::all_close_f(a_grad, vector<float>{0.1966, 0.2500, 0.1966}));
}

TEST(nnfusion_pass_autodiff, roll)
{
    auto model = frontend::load_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/roll.onnx"));

    // a
    auto a = std::vector<float>{0.0071, 0.6307, 0.1928, 0.0560, 0.3107, 0.6996,
                                0.2634, 0.0424, 0.8861, 0.6602, 0.4602, 0.8086,

                                0.8890, 0.6763, 0.1236, 0.9558, 0.9548, 0.7322,
                                0.3866, 0.6383, 0.6436, 0.7410, 0.9749, 0.2272,

                                0.2682, 0.5842, 0.1159, 0.1814, 0.1222, 0.0115,
                                0.4290, 0.1684, 0.1546, 0.3061, 0.2491, 0.7539,

                                0.7455, 0.1327, 0.8785, 0.2097, 0.4326, 0.7425,
                                0.3132, 0.6314, 0.5647, 0.9459, 0.1351, 0.2648};
    auto a_out = vector<float>{0.8890, 0.6763, 0.1236, 0.9558, 0.9548, 0.7322,
                               0.3866, 0.6383, 0.6436, 0.7410, 0.9749, 0.2272,

                               0.0071, 0.6307, 0.1928, 0.0560, 0.3107, 0.6996,
                               0.2634, 0.0424, 0.8861, 0.6602, 0.4602, 0.8086,

                               0.7455, 0.1327, 0.8785, 0.2097, 0.4326, 0.7425,
                               0.3132, 0.6314, 0.5647, 0.9459, 0.1351, 0.2648,

                               0.2682, 0.5842, 0.1159, 0.1814, 0.1222, 0.0115,
                               0.4290, 0.1684, 0.1546, 0.3061, 0.2491, 0.7539};

    auto raw_backward_inputs = make_shared<vector<vector<float>>>();
    raw_backward_inputs->push_back(a_out);
    build_backward_graph(model, raw_backward_inputs);

    RawInputs raw_inputs;

    raw_inputs.emplace_back(convert_to_raw(a));
    RawOutputs raw_outputs{mixed_type_execute(model, raw_inputs, "NNFusion")};
    vector<float> out{convert_from_raw<float>(raw_outputs.at(0))};
    vector<float> a_grad{convert_from_raw<float>(raw_outputs.at(1))};

    EXPECT_TRUE(test::all_close_f(out, a_out));
    EXPECT_TRUE(test::all_close_f(a_grad, a));
}
