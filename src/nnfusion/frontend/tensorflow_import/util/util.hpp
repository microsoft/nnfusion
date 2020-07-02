//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This macro is use to determine if a compiler in use:
//    1. In editor: Use the protobuf files in proto/ for code completion
//    2. In compiling: Use auto-generated probobuf file, Read proto/CmakeLists.txt
//       for details.
#ifdef __cplusplus
#include "graph.pb.h"
#else
#include "../proto/graph.pb.h"
#endif

#include "../tensorflow_base.hpp"

#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/util/util.hpp"

#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace
    {
        std::vector<std::vector<char>>
            get_node_outputs(std::shared_ptr<GNode> gnode, int depth = 0, int arg_idx = 0)
        {
            LOG(INFO) << "[" << depth << ":" << arg_idx
                      << "] Working for node: " << gnode->get_name();
            static std::map<std::shared_ptr<GNode>, std::vector<std::vector<char>>> dict;
            auto it = dict.find(gnode);
            if (it != dict.end())
                return it->second;
            if (gnode->get_op_type() == "Constant")
            {
                auto const_op = std::dynamic_pointer_cast<op::Constant>(gnode->get_op_ptr());
                std::vector<char> one(const_op->get_data_size());
                memcpy(one.data(), const_op->get_data_ptr(), one.size());
                for (int i = 0; i < std::min(10LU, one.size()); ++i)
                    LOG(INFO) << one[i];
                puts("...");
                auto& it = dict[gnode];
                it.push_back(std::move(one));
                CHECK(one.size() == 0);
                return it;
            }
            std::vector<std::vector<char>> _inputs, _outputs;
            int arg_cnt = 0;
            for (auto in_edge : gnode->get_in_edges())
            {
                auto input_node = in_edge->get_src();
                auto outs = get_node_outputs(input_node, depth + 1, arg_cnt++);
                for (auto& out : outs)
                {
                    _inputs.emplace_back(std::move(out));
                    CHECK(out.size() == 0);
                }
            }

            // Prepare runtime backend
            nnfusion::profiler::IProfilingRuntime::Pointer runtime = nullptr;
            std::vector<shared_ptr<const KernelRegistration>> kernel_regs;

            runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
            if (runtime->check_env())
            {
                kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                    gnode->get_op_type(), ROCM_GPU, DT_FLOAT);
                if (kernel_regs.size() == 0)
                    kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                        gnode->get_op_type(), CUDA_GPU, DT_FLOAT);
            }
            else
            {
                runtime = nnfusion::profiler::CudaDefaultRuntime::Runtime();
                CHECK(runtime->check_env());
                kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
                    gnode->get_op_type(), CUDA_GPU, DT_FLOAT);
            }

            bool const_infer_success = false;
            shared_ptr<KernelContext> ctx(new KernelContext(gnode));
            for (auto& kernel_reg : kernel_regs)
            {
                auto kernel = kernel_reg->m_factory(ctx);
                if (!kernel->get_or_emit_source())
                    continue;

                nnfusion::profiler::ProfilingContext::Pointer pctx =
                    make_shared<nnfusion::profiler::ProfilingContext>(kernel);
                pctx->warmup_times = 0;
                pctx->host_times = 1;
                pctx->runtime_times = 1;

                nnfusion::profiler::Profiler prof(runtime, pctx);
                if (!prof.mixed_type_execute(_inputs, _outputs))
                    continue;
                for (int j = 0; j < _outputs.size(); ++j)
                    for (int i = 0; i < std::min(10LU, _outputs[j].size()); ++i)
                        LOG(INFO) << _outputs[j][i];
                puts("...");

                LOG(INFO) << "  For node `" << gnode->get_name()
                          << "`: get runtime output results of size " << _outputs.size();
                const_infer_success = true;
                break;
            }
            return dict[gnode] = _outputs;
        }
    } // namespace

    namespace frontend
    {
        namespace tensorflow_import
        {
            static const int kControlSlot = -1;
            struct TensorId : public std::pair<std::string, int>
            {
                TensorId() {}
                TensorId(const std::string& str, int idx)
                {
                    first = str;
                    second = idx;
                }
                TensorId(const TensorId& id)
                    : TensorId(id.first, id.second)
                {
                }

                const std::string& node() const { return first; }
                int index() const { return second; }
                std::string ToString() const
                {
                    if (second == kControlSlot)
                        return "^" + first;
                    return first + ":" + std::to_string(second);
                }
            };

            // Validates type T for whether it is a supported DataType.
            template <class T>
            struct IsValidDataType;

            // DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
            // constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
            template <class T>
            struct DataTypeToEnum
            {
                static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
            }; // Specializations below

            // EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
            // EnumToDataType<DT_FLOAT>::Type is float.
            template <tensorflow::DataType VALUE>
            struct EnumToDataType
            {
            }; // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                                                            \
    template <>                                                                                    \
    struct DataTypeToEnum<TYPE>                                                                    \
    {                                                                                              \
        static tensorflow::DataType v() { return ENUM; }                                           \
        static constexpr tensorflow::DataType value = ENUM;                                        \
    };                                                                                             \
    template <>                                                                                    \
    struct IsValidDataType<TYPE>                                                                   \
    {                                                                                              \
        static constexpr bool value = true;                                                        \
    };                                                                                             \
    template <>                                                                                    \
    struct EnumToDataType<ENUM>                                                                    \
    {                                                                                              \
        typedef TYPE Type;                                                                         \
    }

            MATCH_TYPE_AND_ENUM(float, tensorflow::DataType::DT_FLOAT);
            MATCH_TYPE_AND_ENUM(double, tensorflow::DataType::DT_DOUBLE);
            MATCH_TYPE_AND_ENUM(int32, tensorflow::DataType::DT_INT32);
            MATCH_TYPE_AND_ENUM(uint32, tensorflow::DataType::DT_UINT32);
            MATCH_TYPE_AND_ENUM(uint16, tensorflow::DataType::DT_UINT16);
            MATCH_TYPE_AND_ENUM(uint8, tensorflow::DataType::DT_UINT8);
            MATCH_TYPE_AND_ENUM(int16, tensorflow::DataType::DT_INT16);
            MATCH_TYPE_AND_ENUM(int8, tensorflow::DataType::DT_INT8);
            MATCH_TYPE_AND_ENUM(std::string, tensorflow::DataType::DT_STRING);
            //MATCH_TYPE_AND_ENUM(complex64, tensorflow::DataType::DT_COMPLEX64);
            //MATCH_TYPE_AND_ENUM(complex128, tensorflow::DataType::DT_COMPLEX128);
            //MATCH_TYPE_AND_ENUM(qint8, tensorflow::DataType::DT_QINT8);
            //MATCH_TYPE_AND_ENUM(quint8, tensorflow::DataType::DT_QUINT8);
            //MATCH_TYPE_AND_ENUM(qint16, tensorflow::DataType::DT_QINT16);
            //MATCH_TYPE_AND_ENUM(quint16, tensorflow::DataType::DT_QUINT16);
            //MATCH_TYPE_AND_ENUM(qint32, tensorflow::DataType::DT_QINT32);
            //MATCH_TYPE_AND_ENUM(bfloat16, tensorflow::DataType::DT_BFLOAT16);
            //MATCH_TYPE_AND_ENUM(Eigen::half, tensorflow::DataType::DT_HALF);
            //MATCH_TYPE_AND_ENUM(ResourceHandle, tensorflow::DataType::DT_RESOURCE);
            //MATCH_TYPE_AND_ENUM(Variant, tensorflow::DataType::DT_VARIANT);
            MATCH_TYPE_AND_ENUM(int64, tensorflow::DataType::DT_INT64);
            MATCH_TYPE_AND_ENUM(uint64, tensorflow::DataType::DT_UINT64);
            MATCH_TYPE_AND_ENUM(bool, tensorflow::DataType::DT_BOOL);

#undef MATCH_TYPE_AND_ENUM

            // Template specialization for both DataTypeToEnum and EnumToDataType.
            // Converts a TensorFlow DataType to an nGraph element::Type. Returns
            // false if the element type is not supported by nGraph
            // Core. Otherwise returns true.
            bool TFDataTypeToNGraphElementType(const tensorflow::DataType tf_dt,
                                               nnfusion::element::Type* ng_et);

            // Converts a TensorFlow TensorShape to an nGraph Shape. Requires that none of
            // the dimension lengths in tf_shape are negative.
            bool TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                            nnfusion::Shape* ng_shape);

            std::shared_ptr<GNode> GetInputNode(const NodeMap& all_ng_nodes,
                                                const tensorflow::NodeDef& node,
                                                size_t input_idx);

            GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes,
                                        const tensorflow::NodeDef& node);

            TensorId ParseTensorName(const std::string& name);

            size_t GetNumElements(const nnfusion::Shape& shape,
                                  const nnfusion::AxisSet& reduction_axes);

            template <typename T, typename VecT = T>
            std::vector<VecT> GetValueFromConstOp(std::shared_ptr<op::Constant> ng_constant_op)
            {
                // the data type of nnfusion::Shape is size_t
                std::vector<VecT> dst_values;
                std::vector<T> values = ng_constant_op->get_vector<T>();
                dst_values.resize(values.size());

                for (size_t i = 0; i < values.size(); i++)
                {
                    dst_values[i] = static_cast<VecT>(values[i]);
                }
                return dst_values;
            }

            template <typename T, typename S>
            void fill_values(std::vector<T>& dst, std::vector<char> src)
            {
                CHECK(src.size() % sizeof(S) == 0);
                dst.resize(src.size() / sizeof(S));
                S* raw_data = (S*)src.data();
                for (int i = 0; i < dst.size(); ++i)
                    dst[i] = raw_data[i];
            }

            template <typename T>
            bool GetValueFromNGraphOp(std::shared_ptr<GNode> gnode, std::vector<T>* values)
            {
                if (gnode->get_op_type() != "Constant")
                {
                    auto outs = get_node_outputs(gnode);
                    CHECK(outs.size() == 1);
                    auto out_type = gnode->get_output_element_type(0);
                    LOG(INFO) << "Asking for Constant value from op-type: " << gnode->get_op_type();
                    LOG(INFO) << "Type of Output Value is " << out_type.c_type_string();

                    if (out_type == nnfusion::element::f32)
                        fill_values<T, float>(*values, outs[0]);
                    else if (out_type == nnfusion::element::f64)
                        fill_values<T, double>(*values, outs[0]);
                    else if (out_type == nnfusion::element::i32)
                        fill_values<T, int>(*values, outs[0]);
                    else if (out_type == nnfusion::element::u32)
                        fill_values<T, unsigned>(*values, outs[0]);
                    else
                    {
                        CHECK_FAIL() << "Unsupport op-type conversion, op-type = " << out_type;
                    }
                    return true;
                }
                auto ng_constant_op = std::dynamic_pointer_cast<op::Constant>(gnode->get_op_ptr());
                auto ng_element_type = gnode->get_output_element_type(0);
                if (ng_element_type == nnfusion::element::f32)
                {
                    *values = GetValueFromConstOp<float, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::f64)
                {
                    *values = GetValueFromConstOp<double, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i8)
                {
                    *values = GetValueFromConstOp<int8, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i16)
                {
                    *values = GetValueFromConstOp<int16, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i32)
                {
                    *values = GetValueFromConstOp<int32, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::i64)
                {
                    *values = GetValueFromConstOp<int64, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u8)
                {
                    *values = GetValueFromConstOp<uint8, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u16)
                {
                    *values = GetValueFromConstOp<uint16, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u32)
                {
                    *values = GetValueFromConstOp<uint32, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::u64)
                {
                    *values = GetValueFromConstOp<uint64, T>(ng_constant_op);
                }
                else if (ng_element_type == nnfusion::element::boolean)
                {
                    *values = GetValueFromConstOp<bool, T>(ng_constant_op);
                }
                else
                {
                    return false;
                }
                return true;
            }

// The ... is to allow the caller to inject some value validation code.  Use
// just ; if no additional validation code is needed.
#define DEFINE_GET_ATTR(TYPE, FIELD, ATTR_TYPE, APPEND_OP, CAST, ...)                              \
    inline bool GetNodeAttr(                                                                       \
        const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,              \
        std::string name,                                                                          \
        TYPE& value)                                                                               \
    {                                                                                              \
        auto attr = attrs.find(name);                                                              \
        if (attr == attrs.end())                                                                   \
            return false;                                                                          \
        const auto& v = attr->second.FIELD() __VA_ARGS__;                                          \
        value = CAST;                                                                              \
        return true;                                                                               \
    }                                                                                              \
    inline bool GetNodeAttr(                                                                       \
        const ::google::protobuf::Map<::std::string, ::tensorflow::AttrValue>& attrs,              \
        std::string name,                                                                          \
        std::vector<TYPE>& value)                                                                  \
    {                                                                                              \
        auto attr = attrs.find(name);                                                              \
        if (attr == attrs.end())                                                                   \
            return false;                                                                          \
        for (const auto& v : attr->second.list().FIELD())                                          \
        {                                                                                          \
            __VA_ARGS__;                                                                           \
            value.APPEND_OP(CAST);                                                                 \
        }                                                                                          \
        return true;                                                                               \
    }

            DEFINE_GET_ATTR(std::string, s, "string", emplace_back, v, ;)
            DEFINE_GET_ATTR(int64, i, "int", emplace_back, v, ;)
            DEFINE_GET_ATTR(int32, i, "int", emplace_back, static_cast<int32>(v), ;)
            DEFINE_GET_ATTR(float, f, "float", emplace_back, v, ;)
            // std::vector<bool> specialization does not have emplace_back until
            // c++14, so we have to use push_back (see
            // http://en.cppreference.com/w/cpp/container/vector/emplace_back)
            DEFINE_GET_ATTR(bool, b, "bool", push_back, v, ;)
            DEFINE_GET_ATTR(tensorflow::DataType,
                            type,
                            "type",
                            emplace_back,
                            static_cast<tensorflow::DataType>(v),
                            ;);
#undef DEFINE_GET_ATTR

            template <size_t a, size_t b, size_t c, size_t d>
            std::shared_ptr<GNode> Reshape(const std::shared_ptr<GNode> old_gnode)
            {
                static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                              "Number of dimensions cannot exceed 4");
                static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                              "Dimensions indices cannot be equal");
                CHECK(old_gnode->get_output_size() == 1);
                auto& s = old_gnode->get_output_shape(0);
                nnfusion::Shape reshaped_shape{s[a], s[b], s[c], s[d]};

                auto reshape_op = std::make_shared<nnfusion::op::Reshape>(
                    nnfusion::AxisVector{a, b, c, d}, reshaped_shape);
                auto reshape_gnode = std::make_shared<GNode>(reshape_op, GNodeVector({old_gnode}));
                reshape_op->revalidate_and_infer_types(reshape_gnode->shared_from_this());

                return reshape_gnode;
            }

            namespace detail
            {
                template <typename T>
                static inline void NhwcToNGraph(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[1];
                    dst[1] = src[2];
                }

                static inline std::shared_ptr<GNode>
                    NhwcToNGraph(const std::shared_ptr<GNode> old_gnode)
                {
                    return Reshape<0, 3, 1, 2>(old_gnode);
                }

                template <typename T>
                static inline void NchwToNGraph(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[2];
                    dst[1] = src[3];
                }

                template <typename T>
                static inline void NhwcToNchw(const std::vector<T>& src, std::vector<size_t>& dst)
                {
                    dst[0] = src[0];
                    dst[1] = src[3];
                    dst[2] = src[1];
                    dst[3] = src[2];
                }
            } // namespace detail

            static inline std::shared_ptr<GNode>
                BatchToNGraph(bool is_nhwc, const std::shared_ptr<GNode> input_gnode)
            {
                if (is_nhwc)
                {
                    return detail::NhwcToNGraph(input_gnode);
                }
                else
                {
                    return nullptr;
                }
            }

            template <typename T>
            static inline void BatchedOpParamToNGraph(bool is_nhwc,
                                                      const std::vector<T>& src,
                                                      std::vector<size_t>& dst)
            {
                if (is_nhwc)
                {
                    detail::NhwcToNGraph(src, dst);
                }
                else
                {
                    detail::NchwToNGraph(src, dst);
                }
            }

            template <typename T>
            static inline void BatchedOpParamReshape(bool is_nhwc,
                                                     const std::vector<T>& src,
                                                     std::vector<size_t>& dst)
            {
                if (is_nhwc)
                {
                    detail::NhwcToNchw(src, dst);
                }
                else
                {
                    dst = src;
                }
            }

            static inline std::shared_ptr<GNode>
                BatchToTensorflow(bool is_nhwc, const std::shared_ptr<GNode> old_gnode)
            {
                if (!is_nhwc)
                {
                    return nullptr;
                }
                return Reshape<0, 2, 3, 1>(old_gnode);
            }

            template <typename T>
            static inline void MakePadding(const std::string& tf_padding_type,
                                           const nnfusion::Shape& ng_image_shape,
                                           const nnfusion::Shape& ng_kernel_shape,
                                           const nnfusion::Strides& ng_strides,
                                           T& ng_padding_below,
                                           T& ng_padding_above)
            {
                if (tf_padding_type == "SAME")
                {
                    for (size_t i = 0; i < 2; i++)
                    {
                        size_t image_size = ng_image_shape[i];
                        size_t filter_shape = ng_kernel_shape[i];
                        size_t filter_stride = ng_strides[i];

                        int64 padding_needed;
                        if (image_size % filter_stride == 0)
                        {
                            padding_needed = filter_shape - filter_stride;
                        }
                        else
                        {
                            padding_needed = filter_shape - (image_size % filter_stride);
                        }
                        if (padding_needed < 0)
                        {
                            padding_needed = 0;
                        }

                        size_t padding_lhs = padding_needed / 2;
                        size_t padding_rhs = padding_needed - padding_lhs;
                        ng_padding_below[i] = padding_lhs;
                        ng_padding_above[i] = padding_rhs;
                    }
                }
            }

            template <typename T>
            static inline void MakePadding(const std::string& tf_padding_type,
                                           const nnfusion::Shape& ng_image_shape,
                                           const nnfusion::Shape& ng_kernel_shape,
                                           const nnfusion::Strides& ng_strides,
                                           const nnfusion::Shape& ng_dilations,
                                           T& ng_padding_below,
                                           T& ng_padding_above)
            {
                nnfusion::Shape ng_dilation_kernel_shape{
                    (ng_kernel_shape[0] - 1) * ng_dilations[0] + 1,
                    (ng_kernel_shape[1] - 1) * ng_dilations[1] + 1};

                MakePadding(tf_padding_type,
                            ng_image_shape,
                            ng_dilation_kernel_shape,
                            ng_strides,
                            ng_padding_below,
                            ng_padding_above);
            }

            static inline bool CheckAxisDimInRange(std::vector<int64> axes, size_t rank)
            {
                for (auto i : axes)
                {
                    if (i < (int)-rank || i >= (int)rank)
                    {
                        LOG(ERROR) << "Axis Dimension is out of range. Got " << i
                                   << ", should be in range [-" << rank << ", " << rank << ")";
                        return false;
                    }
                }
                return true;
            }

        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
