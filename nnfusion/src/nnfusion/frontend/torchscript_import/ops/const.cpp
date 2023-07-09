//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include "const.hpp"
#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            bool isNoneConst(const std::shared_ptr<op::Op> n) { return n == nullptr; }
            std::shared_ptr<op::Op> createNoneConst() { return nullptr; }
            std::shared_ptr<op::Op> MakeConstOp(at::Tensor t)
            {
                if (!t.is_contiguous())
                {
                    t = t.contiguous(c10::MemoryFormat::Contiguous);
                }

                auto t_type = std::string(t.dtype().name());
                auto t_shape = t.sizes();
                auto ng_shape = nnfusion::Shape(t_shape.begin(), t_shape.end());

                if (t_type == "double")
                {
                    // TODO: cast double to float, since double is not supported now
                    auto data_p = t.data_ptr<double>();
                    std::vector<double> values(data_p, data_p + t.numel());
                    std::vector<float> cast_values(values.begin(), values.end());
                    return std::make_shared<op::Constant>(
                        nnfusion::element::f32, // workaround, convert double to f32
                        nnfusion::Shape(t_shape.begin(), t_shape.end()),
                        cast_values);
                }
                else if (t_type == "float")
                {
                    auto data_p = t.data_ptr<float>();
                    std::vector<float> values(data_p, data_p + t.numel());
                    return std::make_shared<op::Constant>(
                        nnfusion::element::f32,
                        nnfusion::Shape(t_shape.begin(), t_shape.end()),
                        values);
                }
                else if (t_type == "long")
                {
                    auto data_p = t.data_ptr<int64>();
                    std::vector<int64> values(data_p, data_p + t.numel());
                    return std::make_shared<op::Constant>(
                        nnfusion::element::i64,
                        nnfusion::Shape(t_shape.begin(), t_shape.end()),
                        values);
                }
                else
                {
                    NNFUSION_CHECK(false) << "Unsupport const tensor type: " << t_type;
                }

                return nullptr;
            }

            // template <typename T, typename ShapeType, typename VecT = T>
            // std::shared_ptr<op::Op> MakeConstOp(ShapeType shape,
            //                                     const std::vector<VecT>& values)
            // {
            //     const nnfusion::element::Type et = nnfusion::element::from<T>();
            //     auto ret = std::make_shared<op::Constant>(et, shape, values);
            //     return ret;
            // }

            std::shared_ptr<op::Op> MakeConstOp(TNodePtr n)
            {
                if (!n->hasAttributeS("value"))
                {
                    // torchscript "None" type, ideally there should be a "None" ngraph node, as a workaround, we return null ptr
                    return createNoneConst();
                }
                auto attr_kind = n->kindOfS("value");
                if (attr_kind == torch::jit::AttributeKind::t)
                {
                    auto v = n->t(c10::Symbol::attr("value"));
                    return MakeConstOp(v);
                }

                std::string const_value;
                // TODO: remove swith, make it as template
                switch (attr_kind)
                {
                case torch::jit::AttributeKind::i:
                {
                    auto v = n->i(c10::Symbol::attr("value"));
                    std::stringstream ss;
                    ss << v;
                    ss >> const_value;
                    break;
                }
                case torch::jit::AttributeKind::f:
                {
                    auto v = n->f(c10::Symbol::attr("value"));
                    std::stringstream ss;
                    ss << v;
                    ss >> const_value;
                    break;
                }
                case torch::jit::AttributeKind::s:
                {
                    auto v = n->s(c10::Symbol::attr("value"));
                    std::stringstream ss;
                    ss << v;
                    ss >> const_value;
                    break;
                }
                default:
                {
                    NNFUSION_CHECK(false)
                        << "Currently we only support convert i/f/s/t constant, but "
                        << torch::jit::toString(attr_kind) << " found";
                }
                }

                auto v_type = n->output()->type()->kind();
                nnfusion::element::Type n_type;
                if (v_type == c10::TypeKind::TensorType)
                {
                    auto scalar_type = n->output()->type()->cast<c10::TensorType>()->scalarType();
                    NNFUSION_CHECK(scalar_type.has_value());
                    NNFUSION_CHECK(ScalarTypeToNGraphElementType(*scalar_type, &n_type));
                }
                else
                {
                    NNFUSION_CHECK(TypeKindToNGraphElementType(v_type, &n_type));
                }

                // Must explict construct Shape
                std::shared_ptr<op::Constant> out_node;
                if (v_type == c10::TypeKind::StringType)
                {
                    std::vector<char> const_value_chars(const_value.begin(), const_value.end());
                    out_node = std::make_shared<op::Constant>(
                        n_type, nnfusion::Shape{const_value_chars.size()}, const_value_chars);
                }
                else
                {
                    out_node = std::make_shared<op::Constant>(
                        n_type, nnfusion::Shape{}, std::vector<std::string>{const_value});
                }
                return out_node;
            }

            bool constEqual(const GNodePtr lhs, const GNodePtr rhs)
            {
                if (!lhs->is_constant() || !rhs->is_constant() ||
                    lhs->get_element_type() != rhs->get_element_type() ||
                    lhs->get_shape() != rhs->get_shape())
                {
                    return false;
                }
                auto lhs_op = std::dynamic_pointer_cast<op::Constant>(lhs->get_op_ptr());
                auto rhs_op = std::dynamic_pointer_cast<op::Constant>(rhs->get_op_ptr());

                return lhs_op->get_vector<int64>() ==
                       rhs_op->get_vector<int64>(); //TODO: remove int64, make it a template func
            }

            // template std::shared_ptr<op::Op> MakeConstOp<int64>(nnfusion::Shape,
            //                                                     const std::vector<int64>&);
            // template std::shared_ptr<op::Op> MakeConstOp<bool>(nnfusion::Shape,
            //                                                    const std::vector<bool>&);
        } // namespace torchscript_import
    }     // namespace frontend
} // namespace nnfusion
