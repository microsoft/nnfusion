//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include "util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            bool TFDataTypeToNNFusionElementType(const tensorflow::DataType tf_dt,
                                                 nnfusion::element::Type* ng_et)
            {
                switch (tf_dt)
                {
                case tensorflow::DataType::DT_HALF: *ng_et = nnfusion::element::f16; break;
                case tensorflow::DataType::DT_FLOAT: *ng_et = nnfusion::element::f32; break;
                case tensorflow::DataType::DT_DOUBLE: *ng_et = nnfusion::element::f64; break;
                case tensorflow::DataType::DT_INT8: *ng_et = nnfusion::element::i8; break;
                case tensorflow::DataType::DT_INT16: *ng_et = nnfusion::element::i16; break;
                case tensorflow::DataType::DT_INT32: *ng_et = nnfusion::element::i32; break;
                case tensorflow::DataType::DT_INT64: *ng_et = nnfusion::element::i64; break;
                case tensorflow::DataType::DT_UINT8: *ng_et = nnfusion::element::u8; break;
                case tensorflow::DataType::DT_UINT16: *ng_et = nnfusion::element::u16; break;
                case tensorflow::DataType::DT_UINT32: *ng_et = nnfusion::element::u32; break;
                case tensorflow::DataType::DT_UINT64: *ng_et = nnfusion::element::u64; break;
                case tensorflow::DataType::DT_BOOL: *ng_et = nnfusion::element::boolean; break;
                case tensorflow::DataType::DT_STRING: *ng_et = nnfusion::element::character; break;
                case tensorflow::DataType::DT_QINT8: *ng_et = nnfusion::element::i8; break;
                case tensorflow::DataType::DT_QUINT8: *ng_et = nnfusion::element::u8; break;
                default: return false;
                }
                return true;
            }

            bool TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                            nnfusion::Shape* ng_shape)
            {
                for (int i = 0; i < tf_shape.dim_size(); i++)
                {
                    if (tf_shape.dim(i).size() < 0)
                    {
                        return false;
                    }
                }

                *ng_shape = nnfusion::Shape(tf_shape.dim_size());
                for (int i = 0; i < tf_shape.dim_size(); i++)
                {
                    (*ng_shape)[i] = tf_shape.dim(i).size();
                }

                return true;
            }

            std::shared_ptr<GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                const tensorflow::NodeDef& node,
                                                size_t input_idx)
            {
                TensorId input_tensor(ParseTensorName(node.input(input_idx)));
                std::shared_ptr<GNode> result = nullptr;
                try
                {
                    result = all_ng_nodes.at(input_tensor.first).at(input_tensor.second);
                }
                catch (const std::out_of_range&)
                {
                    NNFUSION_CHECK_FAIL() << "Input Ngraph op not found for "
                                          << node.input(input_idx);
                }
                return result;
            }

            GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                        const tensorflow::NodeDef& node)
            {
                GNodeVector nodes;
                for (size_t i = 0; i < node.input_size(); i++)
                {
                    nodes.push_back(GetInputNode(all_ng_nodes, node, i));
                }
                return nodes;
            }

            size_t GetNumElements(const nnfusion::Shape& shape,
                                  const nnfusion::AxisSet& reduction_axes)
            {
                size_t N = 1;
                for (auto a : reduction_axes)
                {
                    N *= shape[a];
                }
                return N;
            }

            TensorId ParseTensorName(const std::string& name)
            {
                // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
                // the end of the string, skipping over a run of digits.  If we hit a ':'
                // character, then we know we are in the 'name:digits' regime.  Otherwise, we
                // see if the name starts with '^', indicating a control edge. If we find
                // neither ':' nor '^' characters, the output index is implicitly 0, and the
                // whole name string forms the first part of the tensor name.
                const char* base = name.data();
                const char* p = base + name.size() - 1;
                unsigned int index = 0;
                unsigned int mul = 1;
                while (p > base && (*p >= '0' && *p <= '9'))
                {
                    index += ((*p - '0') * mul);
                    mul *= 10;
                    p--;
                }
                TensorId id;
                if (p > base && *p == ':' && mul > 1)
                {
                    id.first = name.substr(0, p - base);
                    id.second = index;
                }
                else if (name[0] == '^')
                {
                    // Control edge
                    id.first = name.substr(1);
                    id.second = kControlSlot;
                }
                else
                {
                    id.first = name;
                    id.second = 0;
                }
                return id;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
