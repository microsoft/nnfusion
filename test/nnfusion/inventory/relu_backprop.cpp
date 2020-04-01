// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::ReluBackprop
 * \author generated by script
 */

#include "../test_util/common.hpp"
#include "nnfusion/core/operators/parameter.hpp"
#include "nnfusion/core/operators/relu.hpp"

using namespace ngraph;

namespace nnfusion
{
    namespace test
    {
        template <typename T, size_t N>
        using NDArray = nnfusion::test::NDArrayay<T, N>;
    }

    namespace inventory
    {
        template <>
        shared_ptr<op::ReluBackprop> create_object<op::ReluBackprop, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto shape_a = Shape{2, 5};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
                auto relu = make_shared<op::ReluBackprop>(A, delta_val);
                return relu;
            }
            case 1:
            {
                auto shape_a = Shape{2, 2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto delta_val = make_shared<op::Parameter>(element::f32, shape_a);
                auto relu = make_shared<op::ReluBackprop>(A, delta_val);
                return relu;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::ReluBackprop, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5};
                vector<float> delta = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), delta.begin(), delta.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a =
                    vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1};
                vector<float> delta =
                    vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), delta.begin(), delta.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::ReluBackprop, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};
                vector<float> result = expected;
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};
                vector<float> result = expected;
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}