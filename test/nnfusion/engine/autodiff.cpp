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

    void build_backward_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
    {
        FLAGS_fautodiff = true;
        FLAGS_ftraining_optimizer = "{\"optimizer\": \"SGD\", \"learning_rate\": 0.1}";
        auto ad_pass = nnfusion::pass::graph::AutodiffPass();
        ad_pass.run_on_graph(graph);
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
