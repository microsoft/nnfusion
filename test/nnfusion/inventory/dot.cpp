// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Unit tests for ir::Dot
 * \author generated by script
 */

#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace test
    {
        template <typename T, size_t N>
        using NDArray = nnfusion::test::NDArray<T, N>;
    }

    namespace inventory
    {
        template <>
        shared_ptr<graph::GNode> create_object<op::Dot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{0};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{0, 2};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 2:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{4};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 3:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 4:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 5:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{4, 2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{3, 4};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 6:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{2, 2, 2};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 7:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 8:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 9:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{4, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{3};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 10:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{4, 4};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{4};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 11:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 4, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_b{3, 4, 2};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Dot>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Dot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{};
                vector<float> b = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = vector<float>{1};
                vector<float> b = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> a = vector<float>{2, 4, 8, 16};
                vector<float> b = vector<float>{1, 2, 4, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                vector<float> b = vector<float>{5, 6, 7, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
                vector<float> b = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> a = vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
                vector<float> b = vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> a = vector<float>{6};
                vector<float> b = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
                vector<float> b = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> a = vector<float>{8};
                vector<float> b = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                vector<float> b = vector<float>{17, 18, 19};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> a =
                    vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
                vector<float> b = vector<float>{17, 18, 19, 20};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> a_data{6,  61, 2, 3, 5, 21, 75, 23, 23, 0, 23, 2,
                                     35, 67, 1, 2, 9, 16, 2,  3,  6,  1, 8,  0};
                vector<float> a = a_data;
                vector<float> b_data{9, 1,  4,  6, 3, 5, 1, 36, 7, 3, 5, 0,
                                     1, 20, 35, 2, 1, 0, 1, 25, 3, 6, 7, 8};
                vector<float> b = b_data;
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Dot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> result = vector<float>{170};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> result = vector<float>{19, 22, 43, 50};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> result =
                    vector<float>{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> result = vector<float>{
                    20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134, 155, 128, 158, 188, 218,
                    164, 203, 242, 281, 200, 248, 296, 344, 236, 293, 350, 407, 272, 338, 404, 470};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> result = vector<float>{6, 12, 18, 24, 30, 36, 42, 48};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> result = vector<float>{6, 12, 18, 24, 30, 36, 42, 48};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> result = vector<float>{48};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> result = vector<float>{110, 272, 434, 596};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> result = vector<float>{190, 486, 782, 1078};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> result = vector<float>{
                    483,  189, 331, 86,  85,  1262, 2155, 354, 83,  18,   58,   543,  77,
                    241,  325, 286, 859, 144, 438,  1025, 317, 973, 1041, 2930, 163,  69,
                    117,  50,  29,  472, 819, 62,   785,  236, 476, 235,  175,  1521, 2387,
                    1402, 97,  29,  69,  412, 63,   286,  429, 218, 45,   11,   29,   162,
                    27,   106, 149, 126, 65,  25,   44,   6,   11,  165,  281,  52};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    } // namespace inventory
} // namespace nnfusion