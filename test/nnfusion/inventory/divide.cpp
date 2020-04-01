// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Divide
 * \author generated by script
 */

#include "nnfusion/core/operators/divide.hpp"
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
        shared_ptr<op::Divide> create_object<op::Divide, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Divide>(A, B);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Divide, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{2, 4, 8, 16};
                vector<float> b = vector<float>{1, 2, 4, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Divide, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{2, 2, 2, 2};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}