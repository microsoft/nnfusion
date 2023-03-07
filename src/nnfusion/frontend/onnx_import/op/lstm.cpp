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

#include "../util/util.hpp"
#include "attention.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
// #include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "../util/broadcasting.hpp"
#include "util/reshape.hpp"

DECLARE_bool(fantares_mode);
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateLstmSingleOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // now only support 7 inputs [X, W, R, B, seq_len, init_h, init_c]
                    GNodeIndexVector input_indexes;
                    for (int i = 0; i < node_proto.input_size(); i++)
                    {
                        if (i != 4)
                        {
                            auto input_desc = GetInputIndex(all_ng_nodes, node_proto, i);
                            input_indexes.push_back(input_desc);
                        }
                    }
                    Node node(node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    // unsupported attrs: activation related, clip
                    myConfig["direction"] =
                        node.get_attribute_value<std::string>("direction", "forward");
                    myConfig["hidden_size"] = node.get_attribute_value<int64_t>("hidden_size", 0);
                    myConfig["input_forget"] = node.get_attribute_value<int64_t>("input_forget", 0);

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Lstm", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes, 1);

                    return {{node_proto.output(0), generic_gnode, 0}};
                }

                std::vector<std::shared_ptr<graph::GNode>>
                    forwardLSTM(std::shared_ptr<nnfusion::graph::Graph> m_graph,
                                std::shared_ptr<graph::GNode> X,
                                std::shared_ptr<graph::GNode> W,
                                std::shared_ptr<graph::GNode> R,
                                std::shared_ptr<graph::GNode> B,
                                std::shared_ptr<graph::GNode> H0,
                                std::shared_ptr<graph::GNode> C0,
                                std::shared_ptr<graph::GNode> P,
                                bool reverse = false,
                                size_t max_concat_size = std::numeric_limits<std::size_t>::max())
                {
                    auto X_shape = X->get_shape();   // [seq_len, batch, input_size]
                    auto W_shape = W->get_shape();   // [4 * hidden_size, input_size]
                    auto R_shape = R->get_shape();   // [4 * hidden_size, hidden_size]
                    auto B_shape = B->get_shape();   // [8 * hidden_size]
                    auto H0_shape = H0->get_shape(); // [batch_size, hidden_size]
                    auto C0_shape = C0->get_shape(); // [batch_size, hidden_size]
                    // auto P_shape = P->get_shape();   // [batch_size, 3 * hidden_size]
                    size_t seq_len = X_shape.at(0);
                    size_t batch_size = X_shape.at(1);
                    size_t input_size = X_shape.at(2);
                    size_t hidden_size = R_shape.at(1);

                    // split inputs
                    auto W_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{input_size, 4 * hidden_size}),
                        {W});
                    auto R_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{hidden_size, 4 * hidden_size}),
                        {R});
                    auto kernel_gnode = m_graph->add_node_and_edge(std::make_shared<op::Concat>(0),
                                                                   {W_gnode, R_gnode});
                    auto W_b = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{0}, Coordinate{4 * hidden_size}),
                        {B});
                    auto R_b = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{4 * hidden_size},
                                                    Coordinate{8 * hidden_size}),
                        {B});
                    auto bias_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Add>(), {W_b, R_b});
                    bias_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Broadcast>(Shape{batch_size, 4 * hidden_size},
                                                        AxisSet{0}),
                        {bias_gnode});

                    // for in time step
                    auto Ht = H0;
                    auto Ct = C0;
                    GNodeIndexVector H_vec;
                    // std::shared_ptr<graph::GNode> Y_h;
                    for (size_t i = 0; i < seq_len; i++)
                    {
                        // get gates activation
                        size_t x_cur_index = reverse ? seq_len - 1 - i : i;
                        auto slice_op = std::make_shared<op::Slice>(
                            Coordinate{x_cur_index, 0, 0},
                            Coordinate{x_cur_index + 1, batch_size, input_size});
                        auto x =
                            m_graph->add_node_and_edge(slice_op, {X}); // [1, batch, input_size]
                        x = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(AxisVector{0, 1, 2},
                                                          Shape{batch_size, input_size}),
                            {x}); // [batch, input_size]
                        auto xh =
                            m_graph->add_node_and_edge(std::make_shared<op::Concat>(1), {x, Ht});

                        auto gate_inputs = m_graph->add_node_and_edge(std::make_shared<op::Dot>(),
                                                                      {xh, kernel_gnode});
                        gate_inputs = m_graph->add_node_and_edge(std::make_shared<op::Dot>(),
                                                                 {xh, kernel_gnode});
                        gate_inputs = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                                 {gate_inputs, bias_gnode});

                        // gate_inputs = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Reshape>(AxisVector{1, 0},
                        //                                   Shape{4 * hidden_size, batch_size}),
                        //     {gate_inputs});
                        // auto ii = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{0, 0},
                        //                                 Coordinate{hidden_size, batch_size}),
                        //     {gate_inputs});
                        // std::cout << ii->get_output_shape(0) << std::endl;
                        // ii = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Reshape>(AxisVector{1, 0},
                        //                                   Shape{batch_size, hidden_size}),
                        //     {ii});
                        // auto oo = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{hidden_size, 0},
                        //                                 Coordinate{2 * hidden_size, batch_size}),
                        //     {gate_inputs});
                        // std::cout << oo->get_output_shape(0) << std::endl;
                        // oo = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Reshape>(AxisVector{1, 0},
                        //                                   Shape{batch_size, hidden_size}),
                        //     {oo});
                        // auto ff = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{2 * hidden_size, 0},
                        //                                 Coordinate{3 * hidden_size, batch_size}),
                        //     {gate_inputs});
                        // ff = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Reshape>(AxisVector{1, 0},
                        //                                   Shape{batch_size, hidden_size}),
                        //     {ff});
                        // auto cc = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{3 * hidden_size, 0},
                        //                                 Coordinate{4 * hidden_size, batch_size}),
                        //     {gate_inputs});
                        // cc = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Reshape>(AxisVector{1, 0},
                        //                                   Shape{batch_size, hidden_size}),
                        //     {cc});
                        auto ii = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0},
                                                        Coordinate{batch_size, hidden_size}),
                            {gate_inputs});
                        auto oo = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, hidden_size},
                                                        Coordinate{batch_size, 2 * hidden_size}),
                            {gate_inputs});
                        auto ff = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 2 * hidden_size},
                                                        Coordinate{batch_size, 3 * hidden_size}),
                            {gate_inputs});
                        auto cc = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 3 * hidden_size},
                                                        Coordinate{batch_size, 4 * hidden_size}),
                            {gate_inputs});

                        auto it = m_graph->add_node_and_edge(std::make_shared<op::Sigmoid>(), {ii});
                        auto ft = m_graph->add_node_and_edge(std::make_shared<op::Sigmoid>(), {ff});
                        auto ct = m_graph->add_node_and_edge(std::make_shared<op::Tanh>(), {cc});
                        auto ot = m_graph->add_node_and_edge(std::make_shared<op::Sigmoid>(), {oo});

                        auto Ct1 =
                            m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {ft, Ct});
                        auto Ct2 =
                            m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {it, ct});
                        Ct = m_graph->add_node_and_edge(std::make_shared<op::Add>(), {Ct1, Ct2});

                        auto H = m_graph->add_node_and_edge(std::make_shared<op::Tanh>(), {Ct});
                        H = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {ot, H});
                        auto H_reshape = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(AxisVector{0, 1},
                                                          Shape{1, batch_size, hidden_size}),
                            {H}); // [1, batch, hidden_size]

                        H_vec.emplace_back(H_reshape);
                        Ht = H;
                    }
                    if (reverse)
                    {
                        std::reverse(H_vec.begin(), H_vec.end());
                    }

                    // For antares, hierarchically concat result along seq_len
                    while (H_vec.size() > max_concat_size)
                    {
                        GNodeIndexVector new_H_vec;
                        GNodeIndexVector cur_concat_batch;
                        for (size_t i = 0; i < H_vec.size(); i++)
                        {
                            cur_concat_batch.push_back(H_vec[i]);
                            if (cur_concat_batch.size() == max_concat_size)
                            {
                                new_H_vec.emplace_back(m_graph->add_node_and_edge(
                                    std::make_shared<op::Concat>(0), cur_concat_batch));
                                cur_concat_batch.clear();
                            }
                        }
                        if (cur_concat_batch.size() >= 2)
                        {
                            new_H_vec.emplace_back(m_graph->add_node_and_edge(
                                std::make_shared<op::Concat>(0), cur_concat_batch));
                        }
                        else if (cur_concat_batch.size() == 1)
                        {
                            new_H_vec.push_back(cur_concat_batch[0]);
                        }
                        H_vec = new_H_vec;
                    }
                    auto Y = m_graph->add_node_and_edge(std::make_shared<op::Concat>(0), H_vec);
                    // Y: [seq_len, batch, hidden_size]
                    // Ht: [batch, hidden_size]
                    return {Y, Ht, Ct};
                }

                NamedNodeVector TranslateLstmOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // Antares cannot concat too many tensors in sinle IR
                    size_t concat_size_limit = std::numeric_limits<std::size_t>::max();
                    if (FLAGS_fantares_mode)
                        concat_size_limit = 5;

                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto X =
                        GetInputIndex(all_ng_nodes, node_proto, 0); // [seq_len, batch, input_size]
                    auto W = GetInputIndex(
                        all_ng_nodes,
                        node_proto,
                        1); // [num_direction, 4 * hidden_size, input_size], Wi + Wf + Wc + Wo
                    auto R = GetInputIndex(
                        all_ng_nodes,
                        node_proto,
                        2); // [num_direction, 4 * hidden_size, hidden_size], Ri + Rf + Rc + Ro
                    auto X_shape = X.get_shape();
                    size_t seq_len = X_shape.at(0);
                    size_t batch_size = X_shape.at(1);
                    size_t input_size = X_shape.at(2);
                    auto W_shape = W.get_shape();
                    size_t num_directions = W_shape.at(0);
                    size_t hidden_size = W_shape.at(1) / 4;
                    // below are optional input
                    nnfusion::graph::GNodeIndex B, sequence_lens, initial_h, initial_c, P;
                    if (input_indexes.size() >= 4)
                    {
                        B = GetInputIndex(
                            all_ng_nodes,
                            node_proto,
                            3); // [num_direction, 8 * hidden_size], [Wb[ifco], Rb[ifco]]
                    }
                    if (input_indexes.size() >= 5)
                    {
                        sequence_lens = GetInputIndex(all_ng_nodes, node_proto, 4);
                    }
                    if (input_indexes.size() >= 6)
                    {
                        initial_h = GetInputIndex(all_ng_nodes,
                                                  node_proto,
                                                  5); // [num_directions, batch_size, hidden_size]
                    }
                    if (input_indexes.size() >= 7)
                    {
                        initial_c = GetInputIndex(all_ng_nodes,
                                                  node_proto,
                                                  6); // [num_directions, batch_size, hidden_size]
                    }
                    if (input_indexes.size() >= 8)
                    {
                        P = GetInputIndex(all_ng_nodes,
                                          node_proto,
                                          7); // [num_directions, 3 * hidden_size]
                    }

                    if (B.empty())
                    {
                        auto zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Constant>(
                                X.get_element_type(), Shape{}, std::vector<std::string>{"0"}),
                            GNodeIndexVector{});
                        zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(Shape{num_directions, 8 * hidden_size},
                                                            AxisSet{0, 1}),
                            {zero_node});
                        B = GNodeIndex(zero_node);
                    }

                    if (initial_h.empty())
                    {
                        auto zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Constant>(
                                X.get_element_type(), Shape{}, std::vector<std::string>{"0"}),
                            GNodeIndexVector{});
                        zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(
                                Shape{num_directions, batch_size, hidden_size}, AxisSet{0, 1, 2}),
                            {zero_node});
                        initial_h = GNodeIndex(zero_node);
                    }

                    if (initial_c.empty())
                    {
                        auto zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Constant>(
                                X.get_element_type(), Shape{}, std::vector<std::string>{"0"}),
                            GNodeIndexVector{});
                        zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(
                                Shape{num_directions, batch_size, hidden_size}, AxisSet{0, 1, 2}),
                            {zero_node});
                        initial_c = GNodeIndex(zero_node);
                    }

                    NNFUSION_CHECK(P.empty()) << "peepholes of LSTM not supported";
                    // if (P.empty())
                    // {
                    //     auto zero_node = m_graph->add_node_and_edge(
                    //         std::make_shared<op::Constant>(
                    //             X.get_element_type(), Shape{}, std::vector<std::string>{"0"}),
                    //         GNodeIndexVector{});
                    //     zero_node = m_graph->add_node_and_edge(
                    //         std::make_shared<op::Broadcast>(Shape{num_directions, 3 * hidden_size},
                    //                                         AxisSet{0, 1}),
                    //         {zero_node});
                    //     P = GNodeIndex(zero_node);
                    // }

                    NNFUSION_CHECK(sequence_lens.empty()) << "sequence_lens of LSTM not supported";

                    Node node(node_proto);
                    auto activation_alpha = node.get_attribute_value<std::vector<float>>(
                        "activation_alpha", std::vector<float>());
                    auto activation_beta = node.get_attribute_value<std::vector<float>>(
                        "activation_beta", std::vector<float>());
                    auto activations = node.get_attribute_value<std::vector<std::string>>(
                        "activations", std::vector<std::string>());
                    auto clip = node.get_attribute_value<float>("clip", 0);
                    auto direction = node.get_attribute_value<std::string>("direction", "forward");
                    size_t attr_hidden_size = node.get_attribute_value<int64_t>("hidden_size");
                    bool layout = node.get_attribute_value<int64_t>("layout", 0);
                    auto input_forget = node.get_attribute_value<int64_t>("input_forget ", 0);

                    NNFUSION_CHECK(activation_alpha.empty()) << "activation_alpha not supported";
                    NNFUSION_CHECK(activation_beta.empty()) << "activation_beta not supported";
                    NNFUSION_CHECK(activations.empty()) << "activations not supported";
                    NNFUSION_CHECK(clip == 0) << "clip not supported";
                    NNFUSION_CHECK(input_forget == 0) << "input_forget not supported";
                    NNFUSION_CHECK(layout == 0) << "layout not supported";
                    NNFUSION_CHECK(hidden_size == attr_hidden_size) << "hidden_size mismatch";

                    std::shared_ptr<graph::GNode> forward_W, forward_R, forward_B, forward_H,
                        forward_C, forward_P;
                    std::shared_ptr<graph::GNode> reverse_W, reverse_R, reverse_B, reverse_H,
                        reverse_C, reverse_P;
                    if (num_directions == 2)
                    {
                        forward_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0, 0},
                                                        Coordinate{1, 4 * hidden_size, input_size}),
                            {W});
                        forward_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(
                                Coordinate{0, 0, 0}, Coordinate{1, 4 * hidden_size, hidden_size}),
                            {R});
                        forward_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0},
                                                        Coordinate{1, 8 * hidden_size}),
                            {B});
                        forward_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0, 0},
                                                        Coordinate{1, batch_size, hidden_size}),
                            {initial_h});
                        forward_C = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0, 0},
                                                        Coordinate{1, batch_size, hidden_size}),
                            {initial_c});
                        // forward_P = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{0, 0},
                        //                                 Coordinate{1, 3 * hidden_size}),
                        //     {P});

                        reverse_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0, 0},
                                                        Coordinate{2, 4 * hidden_size, input_size}),
                            {W});
                        reverse_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(
                                Coordinate{1, 0, 0}, Coordinate{2, 4 * hidden_size, hidden_size}),
                            {R});
                        reverse_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0},
                                                        Coordinate{2, 8 * hidden_size}),
                            {B});
                        reverse_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0, 0},
                                                        Coordinate{2, batch_size, hidden_size}),
                            {initial_h});
                        reverse_C = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0, 0},
                                                        Coordinate{2, batch_size, hidden_size}),
                            {initial_c});
                        // reverse_P = m_graph->add_node_and_edge(
                        //     std::make_shared<op::Slice>(Coordinate{1, 0},
                        //                                 Coordinate{2, 3 * hidden_size}),
                        //     {P});
                    }
                    else if (direction == "forward")
                    {
                        forward_W = W.gnode;
                        forward_R = R.gnode;
                        forward_B = B.gnode;
                        forward_H = initial_h.gnode;
                        forward_C = initial_c.gnode;
                        // forward_P = P.gnode;
                    }
                    else if (direction == "reverse")
                    {
                        reverse_W = W.gnode;
                        reverse_R = R.gnode;
                        reverse_B = B.gnode;
                        reverse_H = initial_h.gnode;
                        reverse_C = initial_c.gnode;
                        // reverse_P = P.gnode;
                    }

                    std::vector<std::shared_ptr<graph::GNode>> Y_res, Y_h_res, Y_c_res;
                    if (forward_W)
                    {
                        forward_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{4 * hidden_size, input_size}),
                            {forward_W});
                        forward_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{4 * hidden_size, hidden_size}),
                            {forward_R});
                        forward_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{8 * hidden_size}),
                            {forward_B});
                        forward_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {forward_H});
                        forward_C = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {forward_C});
                        std::vector<std::shared_ptr<graph::GNode>> res =
                            forwardLSTM(m_graph,
                                        X.gnode,
                                        forward_W,
                                        forward_R,
                                        forward_B,
                                        forward_H,
                                        forward_C,
                                        forward_P,
                                        false,
                                        concat_size_limit);
                        res[0] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(
                                reshape::get_default_axis_vector(3),
                                Shape{seq_len, 1, batch_size, hidden_size}),
                            {res[0]});
                        res[1] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{1, batch_size, hidden_size}),
                            {res[1]});
                        res[2] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{1, batch_size, hidden_size}),
                            {res[2]});
                        Y_res.push_back(res.at(0));
                        Y_h_res.push_back(res.at(1));
                        Y_c_res.push_back(res.at(2));
                    }
                    if (reverse_W)
                    {
                        reverse_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{4 * hidden_size, input_size}),
                            {reverse_W});
                        reverse_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{4 * hidden_size, hidden_size}),
                            {reverse_R});
                        reverse_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{8 * hidden_size}),
                            {reverse_B});
                        reverse_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {reverse_H});
                        reverse_C = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {reverse_C});
                        std::vector<std::shared_ptr<graph::GNode>> res =
                            forwardLSTM(m_graph,
                                        X.gnode,
                                        reverse_W,
                                        reverse_R,
                                        reverse_B,
                                        reverse_H,
                                        reverse_C,
                                        reverse_P,
                                        true,
                                        concat_size_limit);
                        res[0] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(
                                reshape::get_default_axis_vector(3),
                                Shape{seq_len, 1, batch_size, hidden_size}),
                            {res[0]});
                        res[1] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{1, batch_size, hidden_size}),
                            {res[1]});
                        res[2] = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{1, batch_size, hidden_size}),
                            {res[2]});
                        Y_res.push_back(res.at(0));
                        Y_h_res.push_back(res.at(1));
                        Y_c_res.push_back(res.at(2));
                    }

                    std::shared_ptr<graph::GNode> Y, Y_h, Y_c;
                    if (num_directions == 2)
                    {
                        Y = m_graph->add_node_and_edge(std::make_shared<op::Concat>(1), Y_res);
                        Y_h = m_graph->add_node_and_edge(std::make_shared<op::Concat>(0), Y_h_res);
                        Y_c = m_graph->add_node_and_edge(std::make_shared<op::Concat>(0), Y_c_res);
                    }
                    else
                    {
                        Y = Y_res.at(0);
                        Y_h = Y_h_res.at(0);
                        Y_c = Y_c_res.at(0);
                    }

                    if (node_proto.output_size() == 1)
                    {
                        return {{node_proto.output(0), Y, 0}};
                    }
                    else if (node_proto.output_size() == 2)
                    {
                        return {{node_proto.output(0), Y, 0}, {node_proto.output(1), Y_h, 0}};
                    }
                    else if (node_proto.output_size() == 3)
                    {
                        return {{node_proto.output(0), Y, 0},
                                {node_proto.output(1), Y_h, 0},
                                {node_proto.output(2), Y_c, 0}};
                    }

                    return {};
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
