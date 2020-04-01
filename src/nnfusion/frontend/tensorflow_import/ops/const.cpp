// Microsoft (c) 2019, NNFusion Team

#include "const.hpp"
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
                CHECK(node.op() == "Const");

                if (node.attr().at("dtype").type() != DataTypeToEnum<T>::value)
                {
                    std::stringstream ss;
                    ss << "Invalid data type defined for Const. Defined: "
                       << node.attr().at("dtype").type();
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
                CHECK(0 == tensor_content_size % sizeof(VecT));

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
                        auto dt = node.attr().at("dtype").type();
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
                        // TODO: data type string unsupport now, bert model has string type const op used for assert
                        case tensorflow::DT_STRING:
                            (*values)[i] =
                                0; //(tensor.string_val_size() == 1 ? tensor.string_val()[0]
                                   //   : tensor.string_val()[i]);
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

            template <typename T, typename VecT = T>
            static bool MakeConstOp(const tensorflow::NodeDef& op,
                                    nnfusion::element::Type et,
                                    std::shared_ptr<nnfusion::op::Op>* ng_node)
            {
                std::vector<VecT> const_values;
                tensorflow::TensorShapeProto shape_proto;

                auto ret = ValuesFromConstNode<T, VecT>(op, &shape_proto, &const_values);
                CHECK(ret);

                nnfusion::Shape ng_shape;
                ret = TFTensorShapeToNGraphShape(shape_proto, &ng_shape);
                CHECK(ret);

                *ng_node = std::make_shared<nnfusion::op::Constant>(et, ng_shape, const_values);

                return true;
            }

            const std::map<tensorflow::DataType,
                           std::pair<std::function<bool(const tensorflow::NodeDef&,
                                                        nnfusion::element::Type,
                                                        std::shared_ptr<nnfusion::op::Op>*)>,
                                     const nnfusion::element::Type>>&
                TF_NGRAPH_CONST_MAP()
            {
                static const std::map<
                    tensorflow::DataType,
                    std::pair<std::function<bool(const tensorflow::NodeDef&,
                                                 nnfusion::element::Type,
                                                 std::shared_ptr<nnfusion::op::Op>*)>,
                              const nnfusion::element::Type>>
                    the_map = {
                        {tensorflow::DataType::DT_FLOAT,
                         std::make_pair(MakeConstOp<float>, nnfusion::element::f32)},
                        {tensorflow::DataType::DT_DOUBLE,
                         std::make_pair(MakeConstOp<double>, nnfusion::element::f64)},
                        {tensorflow::DataType::DT_INT8,
                         std::make_pair(MakeConstOp<int8>, nnfusion::element::i8)},
                        {tensorflow::DataType::DT_INT16,
                         std::make_pair(MakeConstOp<int16>, nnfusion::element::i16)},
                        // {tensorflow::DataType::DT_QINT8,
                        //   std::make_pair(MakeConstOp<google::protobuf::qint8>, nnfusion::element::i8)},
                        // {tensorflow::DataType::DT_QUINT16,
                        //   std::make_pair(MakeConstOp<google::protobuf::quint8>, nnfusion::element::u8)},
                        {tensorflow::DataType::DT_INT32,
                         std::make_pair(MakeConstOp<int32>, nnfusion::element::i32)},
                        {tensorflow::DataType::DT_INT64,
                         std::make_pair(MakeConstOp<int64>, nnfusion::element::i64)},
                        {tensorflow::DataType::DT_UINT8,
                         std::make_pair(MakeConstOp<uint8>, nnfusion::element::u8)},
                        {tensorflow::DataType::DT_UINT16,
                         std::make_pair(MakeConstOp<uint16>, nnfusion::element::u16)},
                        {tensorflow::DataType::DT_UINT32,
                         std::make_pair(MakeConstOp<uint32>, nnfusion::element::u32)},
                        {tensorflow::DataType::DT_UINT64,
                         std::make_pair(MakeConstOp<uint64>, nnfusion::element::u64)},
                        {tensorflow::DataType::DT_BOOL,
                         std::make_pair(MakeConstOp<bool, char>, nnfusion::element::boolean)},
                        {tensorflow::DataType::DT_STRING,
                         std::make_pair(MakeConstOp<std::string, char>, nnfusion::element::i32)}};
                // TODO: data type string unsupport now, bert model has string type const op used for assert

                return the_map;
            }

            NamedNodeVector TranslateConstOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
            {
                tensorflow::DataType dtype;
                auto result = GetNodeAttr(node.attr(), "dtype", dtype);
                CHECK(result == true);

                std::shared_ptr<nnfusion::op::Op> ng_node;

                try
                {
                    const auto& func_param = TF_NGRAPH_CONST_MAP().at(dtype);
                    result = func_param.first(node, func_param.second, &ng_node);
                    CHECK(result);
                }
                catch (const std::out_of_range&)
                {
                    CHECK_FAIL_WITH_EXCEPTION(errors::NotSupported)
                        << "Unsupported TensorFlow data type: " << tensorflow::DataType_Name(dtype);
                }
                ng_node->set_name(node.name());
                auto gnode = m_graph->add_node_and_edge(ng_node, GNodeVector({}));
                NamedNodeVector ret{{node.name(), gnode}};

                return ret;
            }
        } // namespace tensorflow_import

    } // namespace frontend

} // namespace nnfusion
