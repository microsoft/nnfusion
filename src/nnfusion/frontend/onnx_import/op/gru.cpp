// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
            namespace
            {
                std::vector<std::shared_ptr<graph::GNode>>
                    forwardGRU(std::shared_ptr<nnfusion::graph::Graph> m_graph,
                               std::shared_ptr<graph::GNode> X,
                               std::shared_ptr<graph::GNode> W,
                               std::shared_ptr<graph::GNode> R,
                               std::shared_ptr<graph::GNode> B,
                               std::shared_ptr<graph::GNode> H0,
                               int linear_before_reset,
                               std::string gate_func,
                               std::string act_func,
                               bool reverse = false,
                               size_t max_concat_size = std::numeric_limits<std::size_t>::max())
                {
                    auto X_shape = X->get_shape();   // [seq_len, batch, input_size]
                    auto W_shape = W->get_shape();   // [3 * hidden_size, input_size]
                    auto R_shape = R->get_shape();   // [3 * hidden_size, hidden_size]
                    auto B_shape = B->get_shape();   // [6 * hidden_size]
                    auto H0_shape = H0->get_shape(); // [batch_size, hidden_size]
                    size_t seq_len = X_shape.at(0);
                    size_t batch_size = X_shape.at(1);
                    size_t input_size = X_shape.at(2);
                    size_t hidden_size = R_shape.at(1);

                    // split inputs
                    auto gates_w_op = std::make_shared<op::Slice>(
                        Coordinate{0, 0}, Coordinate{2 * hidden_size, input_size});
                    auto gates_w = m_graph->add_node_and_edge(gates_w_op, {W});
                    gates_w = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{input_size, 2 * hidden_size}),
                        {gates_w});
                    // gates_w = nnfusion::graph::numpy_transpose(gates_w); // [input_size, 2 * hidden_size]
                    auto w_h = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{2 * hidden_size, 0},
                                                    Coordinate{3 * hidden_size, input_size}),
                        {W});
                    w_h = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{input_size, hidden_size}),
                        {w_h});
                    // w_h = nnfusion::graph::numpy_transpose(w_h); // [input_size, hidden_size]
                    auto gates_r_op = std::make_shared<op::Slice>(
                        Coordinate{0, 0}, Coordinate{2 * hidden_size, hidden_size});
                    auto gates_r = m_graph->add_node_and_edge(gates_r_op, {R});
                    gates_r = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{hidden_size, 2 * hidden_size}),
                        {gates_r});
                    // gates_r = nnfusion::graph::numpy_transpose(gates_r); // [hidden_size, 2 * hidden_size]
                    auto r_h = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{2 * hidden_size, 0},
                                                    Coordinate{3 * hidden_size, hidden_size}),
                        {R});
                    r_h = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(AxisVector{1, 0},
                                                      Shape{hidden_size, hidden_size}),
                        {r_h});
                    // r_h = nnfusion::graph::numpy_transpose(r_h); // [hidden_size, hidden_size]
                    auto gates_b_op_1 =
                        std::make_shared<op::Slice>(Coordinate{0}, Coordinate{2 * hidden_size});
                    auto gates_b_1 = m_graph->add_node_and_edge(gates_b_op_1, {B});
                    auto w_bh = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{2 * hidden_size},
                                                    Coordinate{3 * hidden_size}),
                        {B}); // [hidden_size]
                    auto gates_b_op_2 = std::make_shared<op::Slice>(Coordinate{3 * hidden_size},
                                                                    Coordinate{5 * hidden_size});
                    auto gates_b_2 = m_graph->add_node_and_edge(gates_b_op_2, {B});
                    auto r_bh = m_graph->add_node_and_edge(
                        std::make_shared<op::Slice>(Coordinate{5 * hidden_size},
                                                    Coordinate{6 * hidden_size}),
                        {B}); // [hidden_size]
                    auto gates_b = m_graph->add_node_and_edge(
                        std::make_shared<op::Add>(), {gates_b_1, gates_b_2}); // [2 * hidden_size]

                    // for in time step
                    auto Ht = H0;
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
                        auto gates_1 =
                            m_graph->add_node_and_edge(std::make_shared<op::Dot>(), {x, gates_w});
                        auto gates_2 =
                            m_graph->add_node_and_edge(std::make_shared<op::Dot>(), {Ht, gates_r});
                        auto gates_3 = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(Shape{batch_size, 2 * hidden_size},
                                                            AxisSet{0}),
                            {gates_b});
                        auto gates = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                                {gates_1, gates_2});
                        gates = m_graph->add_node_and_edge(
                            std::make_shared<op::Add>(),
                            {gates, gates_3}); // [batch, 2 * hidden_size]
                        // sigmoid is default gate activation
                        if (gate_func == "Sigmoid")
                        {
                            gates = m_graph->add_node_and_edge(std::make_shared<op::Sigmoid>(),
                                                               {gates});
                        }
                        else if (gate_func == "Tanh")
                        {
                            gates =
                                m_graph->add_node_and_edge(std::make_shared<op::Tanh>(), {gates});
                        }
                        else if (gate_func == "Relu")
                        {
                            gates =
                                m_graph->add_node_and_edge(std::make_shared<op::Relu>(), {gates});
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL() << "Unsuppoted gate funcion: " << gate_func;
                        }
                        slice_op = std::make_shared<op::Slice>(Coordinate{0, 0},
                                                               Coordinate{batch_size, hidden_size});
                        auto z =
                            m_graph->add_node_and_edge(slice_op, {gates}); // [batch, hidden_size]
                        slice_op = std::make_shared<op::Slice>(
                            Coordinate{0, hidden_size}, Coordinate{batch_size, 2 * hidden_size});
                        auto r =
                            m_graph->add_node_and_edge(slice_op, {gates}); // [batch, hidden_size]

                        // get new hidden status
                        auto x_hidden = m_graph->add_node_and_edge(
                            std::make_shared<op::Dot>(), {x, w_h}); // [batch, hidden_size]
                        std::shared_ptr<graph::GNode> h_hidden;
                        if (linear_before_reset == 0)
                        {
                            h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                                  {r, Ht});
                            h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Dot>(),
                                                                  {h_hidden, r_h});
                            auto r_bh_broadcast = m_graph->add_node_and_edge(
                                std::make_shared<op::Broadcast>(Shape{batch_size, hidden_size},
                                                                AxisSet{0}),
                                {r_bh});
                            h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                                  {h_hidden, r_bh_broadcast});
                        }
                        else
                        {
                            h_hidden =
                                m_graph->add_node_and_edge(std::make_shared<op::Dot>(), {Ht, r_h});
                            auto r_bh_broadcast = m_graph->add_node_and_edge(
                                std::make_shared<op::Broadcast>(Shape{batch_size, hidden_size},
                                                                AxisSet{0}),
                                {r_bh});
                            h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                                  {h_hidden, r_bh_broadcast});
                            h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                                  {r, h_hidden});
                        }
                        auto w_bh_broadcast = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(Shape{batch_size, hidden_size},
                                                            AxisSet{0}),
                            {w_bh});
                        h_hidden = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                              {h_hidden, w_bh_broadcast});
                        auto h = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                            {x_hidden, h_hidden});
                        // tanh is default linear activation
                        if (act_func == "Tanh")
                        {
                            h = m_graph->add_node_and_edge(std::make_shared<op::Tanh>(),
                                                           {h}); // [batch, hidden_size]
                        }
                        else if (act_func == "Sigmoid")
                        {
                            h = m_graph->add_node_and_edge(std::make_shared<op::Sigmoid>(),
                                                           {h}); // [batch, hidden_size]
                        }
                        else if (act_func == "Relu")
                        {
                            h = m_graph->add_node_and_edge(std::make_shared<op::Relu>(),
                                                           {h}); // [batch, hidden_size]
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL() << "Unsuppoted activation: " << act_func;
                        }
                        auto one_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Constant>(
                                h->get_element_type(), Shape{}, std::vector<std::string>{"1"}),
                            GNodeIndexVector{});
                        one_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(Shape{batch_size, hidden_size},
                                                            AxisSet{0, 1}),
                            {one_node});
                        auto H_temp1 = m_graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                                                  {one_node, z});
                        H_temp1 = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                             {H_temp1, h});
                        auto H_temp2 =
                            m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {z, Ht});
                        auto H =
                            m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                       {H_temp1, H_temp2}); // [batch, hidden_size]
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
                    return {Y, Ht};
                }
            }
            namespace set_1
            {
                NamedNodeVector TranslateGRUOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    NNFUSION_CHECK_FAIL() << "GRU not support for current op_set";
                    return {};
                }
            }

            namespace set_7
            {
                NamedNodeVector TranslateGRUOp(const onnx::NodeProto& node_proto,
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
                        1); // [num_direction, 3 * hidden_size, input_size], Wz + Wr + Wh
                    auto R = GetInputIndex(
                        all_ng_nodes,
                        node_proto,
                        2); // [num_direction, 3 * hidden_size, hidden_size], Rz + Rr + Rh
                    auto X_shape = X.get_shape();
                    size_t seq_len = X_shape.at(0);
                    size_t batch_size = X_shape.at(1);
                    size_t input_size = X_shape.at(2);
                    auto W_shape = W.get_shape();
                    size_t num_directions = W_shape.at(0);
                    size_t hidden_size = W_shape.at(1) / 3;
                    // below are optional input
                    nnfusion::graph::GNodeIndex B, sequence_lens, initial_h;
                    if (input_indexes.size() >= 4)
                    {
                        B = GetInputIndex(
                            all_ng_nodes,
                            node_proto,
                            3); // [num_direction, 6 * hidden_size], [Wb[zrh], Rb[zrh]]
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

                    if (B.empty())
                    {
                        auto zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Constant>(
                                X.get_element_type(), Shape{}, std::vector<std::string>{"0"}),
                            GNodeIndexVector{});
                        zero_node = m_graph->add_node_and_edge(
                            std::make_shared<op::Broadcast>(Shape{num_directions, 6 * hidden_size},
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

                    NNFUSION_CHECK(sequence_lens.empty()) << "sequence_lens of GRU not supported";

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
                    auto layout = node.get_attribute_value<int64_t>("layout", 0);
                    auto linear_before_reset =
                        node.get_attribute_value<int64_t>("linear_before_reset", 0);

                    NNFUSION_CHECK(activation_alpha.empty()) << "activation_alpha not supported";
                    NNFUSION_CHECK(activation_beta.empty()) << "activation_beta not supported";
                    if (activations.empty())
                    {
                        activations.push_back("Sigmoid");
                        activations.push_back("Tanh");
                        if (num_directions > 1)
                        {
                            activations.push_back("Sigmoid");
                            activations.push_back("Tanh");
                        }
                    }
                    else
                    {
                        NNFUSION_CHECK(activations.size() == 2 * num_directions)
                            << "gru activations size should be equal to 2 * num_directions";
                    }
                    NNFUSION_CHECK(clip == 0) << "clip not supported";
                    NNFUSION_CHECK(layout == 0) << "layout not supported";
                    NNFUSION_CHECK(hidden_size == attr_hidden_size) << "hidden_size mismatch";

                    std::shared_ptr<graph::GNode> forward_W, forward_R, forward_B, forward_H;
                    std::string forward_gate, forward_act;
                    std::shared_ptr<graph::GNode> reverse_W, reverse_R, reverse_B, reverse_H;
                    std::string reverse_gate, reverse_act;

                    if (num_directions == 2)
                    {
                        forward_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0, 0},
                                                        Coordinate{1, 3 * hidden_size, input_size}),
                            {W});
                        forward_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(
                                Coordinate{0, 0, 0}, Coordinate{1, 3 * hidden_size, hidden_size}),
                            {R});
                        forward_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0},
                                                        Coordinate{1, 6 * hidden_size}),
                            {B});
                        forward_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{0, 0, 0},
                                                        Coordinate{1, batch_size, hidden_size}),
                            {initial_h});

                        reverse_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0, 0},
                                                        Coordinate{2, 3 * hidden_size, input_size}),
                            {W});
                        reverse_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(
                                Coordinate{1, 0, 0}, Coordinate{2, 3 * hidden_size, hidden_size}),
                            {R});
                        reverse_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0},
                                                        Coordinate{2, 6 * hidden_size}),
                            {B});
                        reverse_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Slice>(Coordinate{1, 0, 0},
                                                        Coordinate{2, batch_size, hidden_size}),
                            {initial_h});
                        forward_gate = activations[0];
                        forward_act = activations[1];
                        reverse_gate = activations[2];
                        reverse_act = activations[3];
                    }
                    else if (direction == "forward")
                    {
                        forward_W = W.gnode;
                        forward_R = R.gnode;
                        forward_B = B.gnode;
                        forward_H = initial_h.gnode;
                        forward_gate = activations[0];
                        forward_act = activations[1];
                    }
                    else if (direction == "reverse")
                    {
                        reverse_W = W.gnode;
                        reverse_R = R.gnode;
                        reverse_B = B.gnode;
                        reverse_H = initial_h.gnode;
                        reverse_gate = activations[0];
                        reverse_act = activations[1];
                    }

                    std::vector<std::shared_ptr<graph::GNode>> Y_res, Y_h_res;
                    if (forward_W)
                    {
                        forward_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{3 * hidden_size, input_size}),
                            {forward_W});
                        forward_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{3 * hidden_size, hidden_size}),
                            {forward_R});
                        forward_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{6 * hidden_size}),
                            {forward_B});
                        forward_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {forward_H});
                        std::vector<std::shared_ptr<graph::GNode>> res =
                            forwardGRU(m_graph,
                                       X.gnode,
                                       forward_W,
                                       forward_R,
                                       forward_B,
                                       forward_H,
                                       linear_before_reset,
                                       forward_gate,
                                       forward_act,
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
                        Y_res.push_back(res.at(0));
                        Y_h_res.push_back(res.at(1));
                    }
                    if (reverse_W)
                    {
                        reverse_W = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{3 * hidden_size, input_size}),
                            {reverse_W});
                        reverse_R = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{3 * hidden_size, hidden_size}),
                            {reverse_R});
                        reverse_B = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(2),
                                                          Shape{6 * hidden_size}),
                            {reverse_B});
                        reverse_H = m_graph->add_node_and_edge(
                            std::make_shared<op::Reshape>(reshape::get_default_axis_vector(3),
                                                          Shape{batch_size, hidden_size}),
                            {reverse_H});
                        std::vector<std::shared_ptr<graph::GNode>> res =
                            forwardGRU(m_graph,
                                       X.gnode,
                                       reverse_W,
                                       reverse_R,
                                       reverse_B,
                                       reverse_H,
                                       linear_before_reset,
                                       reverse_gate,
                                       reverse_act,
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
                        Y_res.push_back(res.at(0));
                        Y_h_res.push_back(res.at(1));
                    }

                    std::shared_ptr<graph::GNode> Y, Y_h;
                    if (num_directions == 2)
                    {
                        Y = m_graph->add_node_and_edge(std::make_shared<op::Concat>(1), Y_res);
                        Y_h = m_graph->add_node_and_edge(std::make_shared<op::Concat>(0), Y_h_res);
                    }
                    else
                    {
                        Y = Y_res.at(0);
                        Y_h = Y_h_res.at(0);
                    }

                    return {{node_proto.output(0), Y, 0}, {node_proto.output(1), Y_h, 0}};
                }

            } // namespace set_7

            namespace set_14
            {
                using set_7::TranslateGRUOp;
            }

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
