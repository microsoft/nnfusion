// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "const.hpp"
#include "nnfusion/common/type/data_buffer.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "stdint.h"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            template <typename T, typename VecT = T>
            bool ValuesFromConstNode(const tensorflow::NodeDef& node,
                                     tensorflow::TensorShapeProto* const_tensor_shape,
                                     std::vector<VecT>* values)
            {
                NNFUSION_CHECK(node.op() == "Const");
                auto dt = node.attr().at("dtype").type();
                if (dt != DataTypeToEnum<T>::value)
                {
                    std::stringstream ss;
                    ss << "Invalid data type defined for Const. Defined: " << dt;
                    return false;
                }

                // TensorProto represents the content of the tensor in either <type>_val or
                // tensor_content.
                const tensorflow::TensorProto& tensor = node.attr().at("value").tensor();
                // typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
                //     checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

                const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();
                *const_tensor_shape = shape;
                // if (!tensor_values->empty() && tensor.has_tensor_shape())
                // {
                //     // When tensor_shape is set, theoretically the representation of the data
                //     // could be compressed. So, before copying values to the returned vector,
                //     // make sure no compression happens.
                //     if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size())
                //     {
                //         values->insert(values->end(), tensor_values->begin(), tensor_values->end());
                //         return true;
                //     }
                // }

                const auto tensor_content_size = tensor.tensor_content().size();
                NNFUSION_CHECK(0 == tensor_content_size % sizeof(VecT));

                // If tensor_content_size is zero, we'll have to take the values from
                // int_val, float_val, etc.
                if (tensor_content_size == 0)
                {
                    int64_t n_elements = 1;
                    for (size_t i = 0; i < shape.dim_size(); i++)
                    {
                        if (shape.dim(i).size() < 0)
                        {
                            return false;
                            // return errors::InvalidArgument(
                            //     "Const node has empty tensor and an unknown dimension size");
                        }
                        n_elements *= shape.dim(i).size();
                    }

                    values->resize(n_elements);
                    for (size_t i = 0; i < n_elements; i++)
                    {
                        auto& tensor = node.attr().at("value").tensor();
                        switch (dt)
                        {
                        // TODO(amprocte/NGRAPH-2502): there are more element types to support
                        // here
                        case tensorflow::DT_INT32:
                            (*values)[i] = (tensor.int_val_size() == 1 ? tensor.int_val()[0]
                                                                       : tensor.int_val()[i]);
                            break;
                        case tensorflow::DT_INT64:
                            (*values)[i] = (tensor.int64_val_size() == 1 ? tensor.int64_val()[0]
                                                                         : tensor.int64_val()[i]);
                            break;
                        case tensorflow::DT_FLOAT:
                            (*values)[i] = (tensor.float_val_size() == 1 ? tensor.float_val()[0]
                                                                         : tensor.float_val()[i]);
                            break;
                        case tensorflow::DT_BOOL:
                            (*values)[i] = (tensor.bool_val_size() == 1 ? tensor.bool_val()[0]
                                                                        : tensor.bool_val()[i]);
                            break;
                        case tensorflow::DT_DOUBLE:
                            (*values)[i] = (tensor.double_val_size() == 1 ? tensor.double_val()[0]
                                                                          : tensor.double_val()[i]);
                            break;
                        case tensorflow::DT_STRING:
                            if (i > 0)
                            {
                                // TODO: only support one dimension for string type now
                                return false;
                            }
                            values->resize(tensor.string_val()[0].length());
                            std::copy(tensor.string_val()[0].begin(),
                                      tensor.string_val()[0].end(),
                                      values->begin());
                            break;
                        default:
                            return false;
                            // NGRAPH_VLOG(0)
                            //     << "Const node has empty tensor and we don't know how to "
                            //        "handle this element type";
                            // NGRAPH_VLOG(0) << node.DebugString();
                            // NGRAPH_VLOG(0) << shape.DebugString();
                            // return errors::Unimplemented("Encountered unknown element type ",
                            //                              DataType_Name(dt),
                            //                              " on an empty tensor");
                        }
                    }
                }
                else
                {
                    values->resize(tensor_content_size / sizeof(VecT));
                    CopyToArray(tensor.tensor_content(), reinterpret_cast<char*>(values->data()));
                }
                return true;
            }

            bool ValuesFromConstNode(const tensorflow::NodeDef& node,
                                     tensorflow::TensorShapeProto* const_tensor_shape,
                                     DataBuffer* values)
            {
                NNFUSION_CHECK(node.op() == "Const");
                auto dt = node.attr().at("dtype").type();
                // if (dt != DataTypeToEnum<T>::value)
                // {
                //     std::stringstream ss;
                //     ss << "Invalid data type defined for Const. Defined: " << dt;
                //     return false;
                // }

                // TensorProto represents the content of the tensor in either <type>_val or
                // tensor_content.
                const tensorflow::TensorProto& tensor = node.attr().at("value").tensor();
                // typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
                //     checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

                const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();
                *const_tensor_shape = shape;

                const auto tensor_content_size = tensor.tensor_content().size();
                // NNFUSION_LOG(INFO) << values->get_type() << ": tensor_size = " << tensor_content_size << ", type size = " << values->get_type().size();
                NNFUSION_CHECK(0 == tensor_content_size % values->get_type().size());

                int64_t n_elements = 1;
                for (size_t i = 0; i < shape.dim_size(); i++)
                {
                    if (shape.dim(i).size() < 0)
                    {
                        return false;
                        // return errors::InvalidArgument(
                        //     "Const node has empty tensor and an unknown dimension size");
                    }
                    n_elements *= shape.dim(i).size();
                }

                // If tensor_content_size is zero, we'll have to take the values from
                // int_val, float_val, etc.
                if (tensor_content_size == 0)
                {
#define GET_VALUES(type)                                                                           \
    do                                                                                             \
    {                                                                                              \
        const void* dat = nullptr;                                                                 \
        for (size_t i = 0; i < n_elements; ++i)                                                    \
        {                                                                                          \
            if (tensor.type##_val_size() == 1)                                                     \
            {                                                                                      \
                dat = reinterpret_cast<const void*>(&tensor.type##_val()[0]);                      \
            }                                                                                      \
            else                                                                                   \
            {                                                                                      \
                dat = reinterpret_cast<const void*>(&tensor.type##_val()[i]);                      \
            }                                                                                      \
            values->setElement(i, dat);                                                            \
        }                                                                                          \
    } while (0)

                    values->resize(n_elements);
                    auto& tensor = node.attr().at("value").tensor();
                    size_t val_size;
                    if (dt == tensorflow::DT_INT32)
                    {
                        GET_VALUES(int);
                    }
                    else if (dt == tensorflow::DT_INT64)
                    {
                        GET_VALUES(int64);
                    }
                    else if (dt == tensorflow::DT_BOOL)
                    {
                        GET_VALUES(bool);
                    }
                    else if (dt == tensorflow::DT_HALF)
                    {
                        GET_VALUES(half);
                    }
                    else if (dt == tensorflow::DT_FLOAT)
                    {
                        GET_VALUES(float);
                    }
                    else if (dt == tensorflow::DT_DOUBLE)
                    {
                        GET_VALUES(double);
                    }
                    else if (dt == tensorflow::DT_STRING)
                    {
                        values->resize(tensor.string_val()[0].length());
                        auto it = tensor.string_val()[0].begin();
                        for (size_t j = 0; it != tensor.string_val()[0].end(); ++j, ++it)
                        {
                            values->setElement(j, reinterpret_cast<const void*>(&it));
                        }
                    }
                    else
                    {
                        return false;
                    }

#undef GET_VALUES
                }
                else
                {
                    size_t size_tensor = tensor_content_size / values->get_type().size();
                    const char* content = tensor.tensor_content().c_str();
                    if (size_tensor > 1 || size_tensor == n_elements)
                    {
                        values->load(content, size_tensor);
                    }
                    else
                    {
                        for (size_t i = 0; i < n_elements; ++i)
                        {
                            values->setElement(i, content);
                        }
                    }
                }
                return true;
            }

            template <typename T, typename VecT = T>
            static bool MakeConstOp(const tensorflow::NodeDef& op,
                                    nnfusion::element::Type et,
                                    std::shared_ptr<nnfusion::op::Op>* ng_node)
            {
                std::vector<VecT> const_values;
                tensorflow::TensorShapeProto shape_proto;

                auto ret = ValuesFromConstNode<T, VecT>(op, &shape_proto, &const_values);
                NNFUSION_CHECK(ret);

                nnfusion::Shape ng_shape;
                ret = TFTensorShapeToNGraphShape(shape_proto, &ng_shape);
                NNFUSION_CHECK(ret);
                if (et == nnfusion::element::character)
                {
                    NNFUSION_CHECK(ng_shape.size() <= 1)
                        << "For string type constant op, only one dimension support!";
                    *ng_node = std::make_shared<nnfusion::op::Constant>(
                        et, nnfusion::Shape{const_values.size()}, const_values);
                }
                else
                {
                    *ng_node = std::make_shared<nnfusion::op::Constant>(et, ng_shape, const_values);
                }

                return true;
            }

            static bool MakeConstOp(const tensorflow::NodeDef& op,
                                    nnfusion::element::Type et,
                                    std::shared_ptr<nnfusion::op::Op>* ng_node)
            {
                DataBuffer const_values(et);
                tensorflow::TensorShapeProto shape_proto;

                auto ret = ValuesFromConstNode(op, &shape_proto, &const_values);
                NNFUSION_CHECK(ret);

                nnfusion::Shape ng_shape;
                ret = TFTensorShapeToNGraphShape(shape_proto, &ng_shape);
                NNFUSION_CHECK(ret);
                if (et == nnfusion::element::character)
                {
                    NNFUSION_CHECK(ng_shape.size() <= 1)
                        << "For string type constant op, only one dimension support!";
                    *ng_node = std::make_shared<nnfusion::op::Constant>(
                        et, nnfusion::Shape{const_values.size()}, const_values);
                }
                else
                {
                    *ng_node = std::make_shared<nnfusion::op::Constant>(et, ng_shape, const_values);
                }

                return true;
            }

            NamedNodeVector TranslateConstOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                tensorflow::DataType dtype;
                auto result = GetNodeAttr(node.attr(), "dtype", dtype);
                NNFUSION_CHECK(result == true);

                std::shared_ptr<nnfusion::op::Op> ng_node;

                try
                {
                    element::Type type;
                    result = TFDataTypeToNNFusionElementType(dtype, &type);
                    NNFUSION_CHECK(result);
                    result = MakeConstOp(node, type, &ng_node);
                    NNFUSION_CHECK(result);
                }
                catch (const std::out_of_range&)
                {
                    NNFUSION_CHECK_FAIL_WITH_EXCEPTION(errors::NotSupported)
                        << "Unsupported TensorFlow data type: " << tensorflow::DataType_Name(dtype);
                }
                ng_node->set_name(node.name());
                auto gnode = m_graph->add_node_and_edge(ng_node, GNodeVector({}));
                NamedNodeVector ret{{node.name(), gnode}};

                return ret;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
