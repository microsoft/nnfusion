// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Greater
 * \author generated by script
 */

#include "nnfusion/core/operators/greater.hpp"
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
        shared_ptr<op::Greater> create_object<op::Greater, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Greater>(A, B);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Greater, float>(int option)
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
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Greater, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{0, 1, 0, 1, 0, 1, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}