// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>

#include "onnx/onnx-ml.pb.h"

#include "ngraph/type/element_type.hpp"
#include "nnfusion/core/operators/convert.hpp"

#include "cast.hpp"
#include "exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector cast(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    int64_t target_type = node.get_attribute_value<int64_t>("to");
                    element::Type elem_type;

                    switch (target_type)
                    {
                    case onnx::TensorProto_DataType_BOOL: elem_type = element::boolean; break;
                    case onnx::TensorProto_DataType_DOUBLE: elem_type = element::f64; break;
                    case onnx::TensorProto_DataType_FLOAT16:
                    case onnx::TensorProto_DataType_FLOAT: elem_type = element::f32; break;
                    case onnx::TensorProto_DataType_INT8: elem_type = element::i8; break;
                    case onnx::TensorProto_DataType_INT16: elem_type = element::i16; break;
                    case onnx::TensorProto_DataType_INT32: elem_type = element::i32; break;
                    case onnx::TensorProto_DataType_INT64: elem_type = element::i64; break;
                    case onnx::TensorProto_DataType_UINT8: elem_type = element::u8; break;
                    case onnx::TensorProto_DataType_UINT16: elem_type = element::u16; break;
                    case onnx::TensorProto_DataType_UINT32: elem_type = element::u32; break;
                    case onnx::TensorProto_DataType_UINT64: elem_type = element::u64; break;
                    case onnx::TensorProto_DataType_UNDEFINED: elem_type = element::dynamic; break;
                    default: ASSERT_IS_SUPPORTED(node, false) << "unsupported type";
                    }

                    return {std::make_shared<ngraph::op::Convert>(data, elem_type)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
