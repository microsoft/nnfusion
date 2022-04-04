// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Unit tests for ir::Minimum
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
        shared_ptr<graph::GNode> create_object<op::Minimum, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Minimum>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto B_gnode = graph->add_node_and_edge(B, GNodeVector({}));
                auto r = make_shared<op::Minimum>();
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode, B_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Minimum, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
                vector<float> b = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = vector<float>{1, 8, -8, 17, -5, 67635216, 2, 1};
                vector<float> b = vector<float>{1, 2, 4, 8, 0, 18448, 1, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Minimum, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 2, -8, 8, -.5, 0, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = vector<float>{1, 2, -8, 8, -5, 18448, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    } // namespace inventory
} // namespace nnfusion