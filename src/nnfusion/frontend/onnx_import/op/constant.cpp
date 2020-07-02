// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/constant.hpp"
#include "core/node.hpp"
#include "core/tensor.hpp"
#include "ngraph/node_vector.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                namespace
                {
                    template <typename T>
                    inline std::shared_ptr<ngraph::op::Constant>
                        __make_ng_constant(const element::Type& type, const Tensor& tensor)
                    {
                        return std::make_shared<ngraph::op::Constant>(
                            type, tensor.get_shape(), tensor.get_data<T>());
                    }

                    template <Tensor::Type>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant(const Tensor& tensor)
                    {
                        throw error::tensor::unsupported_data_type{tensor};
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::float16>(const Tensor& tensor)
                    {
                        return __make_ng_constant<float>(element::f32, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::float32>(const Tensor& tensor)
                    {
                        return __make_ng_constant<float>(element::f32, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::float64>(const Tensor& tensor)
                    {
                        return __make_ng_constant<double>(element::f64, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::int32>(const Tensor& tensor)
                    {
                        return __make_ng_constant<int32_t>(element::i32, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::int64>(const Tensor& tensor)
                    {
                        return __make_ng_constant<int64_t>(element::i64, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::uint32>(const Tensor& tensor)
                    {
                        return __make_ng_constant<uint32_t>(element::u32, tensor);
                    }

                    template <>
                    inline std::shared_ptr<ngraph::op::Constant>
                        make_ng_constant<Tensor::Type::uint64>(const Tensor& tensor)
                    {
                        return __make_ng_constant<uint64_t>(element::u64, tensor);
                    }

                    inline std::shared_ptr<ngraph::op::Constant> make_constant(const Tensor& tensor)
                    {
#define MAKE_NG_CONSTANT(data_type_)                                                               \
    case data_type_: return make_ng_constant<data_type_>(tensor)

                        switch (tensor.get_type())
                        {
                            MAKE_NG_CONSTANT(Tensor::Type::float16);
                            MAKE_NG_CONSTANT(Tensor::Type::float32);
                            MAKE_NG_CONSTANT(Tensor::Type::float64);
                            MAKE_NG_CONSTANT(Tensor::Type::int32);
                            MAKE_NG_CONSTANT(Tensor::Type::int64);
                            MAKE_NG_CONSTANT(Tensor::Type::uint32);
                            MAKE_NG_CONSTANT(Tensor::Type::uint64);
                        default: throw error::tensor::invalid_data_type{tensor};
                        }
                    }
                }

                NodeVector constant(const onnx_import::Node& node)
                {
                    return {make_constant(node.get_attribute_value<Tensor>("value"))};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
