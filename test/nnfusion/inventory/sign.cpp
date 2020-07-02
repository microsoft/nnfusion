// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
/**
 * \brief Unit tests for ir::Sign
 * \author generated by script
 */

#include "nnfusion/core/operators/sign.hpp"
#include "../test_util/common.hpp"
#include "nnfusion/core/operators/parameter.hpp"

using namespace ngraph;

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
        shared_ptr<op::Sign> create_object<op::Sign, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Sign>(A);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Sign, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Sign, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, -1, 0, -1, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}