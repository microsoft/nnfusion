// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::OneHot
 * \author generated by script
 */

#include "nnfusion/core/operators/one_hot.hpp"
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
        shared_ptr<op::OneHot> create_object<op::OneHot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
                return r;
            }
            case 1:
            {
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
                return r;
            }
            case 2:
            {
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
                return r;
            }
            case 3:
            {
                Shape shape_a{8};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
                return r;
            }
            case 4:
            {
                Shape shape_a{8};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
                return r;
            }
            case 5:
            {
                Shape shape_a{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
                return r;
            }
            case 6:
            {
                Shape shape_a{8};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
                return r;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::OneHot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{2};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = vector<float>{1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> a = vector<float>{0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> a = vector<float>{2, 1, 0, 0, 2, 2, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> a = vector<float>{2, 1, 0, 0, 2, 2, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> a = vector<float>{
                    0, 1, 1, 2, 1, 0, 0, 2, 1,
                };
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> a = vector<float>{2, 1, 0, 0, 2, 2, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::OneHot, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{0, 0, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = vector<float>{0, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> result = vector<float>{1, 0, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> result = vector<float>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
                                                     0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> result = vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                     0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> result = vector<float>{1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1,
                                                     0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> result = vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                                                     0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}