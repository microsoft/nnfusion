// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cuda_cudnn.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::kernels;

std::vector<int> cuda::compute_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }
    return strides;
}

std::string cuda::get_cudnn_datatype(element::Type dtype)
{
    static const std::unordered_map<std::string, std::string> datatype_map{
        {"half", "CUDNN_DATA_HALF"},
        {"float", "CUDNN_DATA_FLOAT"},
        {"double", "CUDNN_DATA_DOUBLE"},
        {"int8_t", "CUDNN_DATA_INT8"},
        {"int32_t", "CUDNN_DATA_INT32"}};
    auto p = datatype_map.find(dtype.c_type_string());
    NNFUSION_CHECK(p != datatype_map.end()) << dtype << " is not supported by cuDNN";

    return p->second;
}

LanguageUnit_p cuda::cudnn_tensor_descriptor_from_shape(const nnfusion::Shape& shape,
                                                        string desc,
                                                        element::Type type)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;
    string data_type = cuda::get_cudnn_datatype(type);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnTensorDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&" << desc << "));\n";

    if (shape.size() < 4)
    {
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(shape[i]);
        }
        lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
           << data_type << ", " << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2]
           << ", " << dimensions[3] << "));\n";
    }
    else if (shape.size() == 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
           << data_type << ", " << static_cast<int>(shape[0]) << ", " << static_cast<int>(shape[1])
           << ", " << static_cast<int>(shape[2]) << ", " << static_cast<int>(shape[3]) << "));\n";
    }
    else
    {
        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            NNFUSION_CHECK(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}\n";
            return ss.str();
        };

        std::vector<int> dimensions(shape.size());
        for (auto i = 0u; i < shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(shape[i]);
        }
        vector<int> strides = cuda::compute_strides(dimensions);

        lu << expand_vector_int(desc + "_dim", dimensions);
        lu << expand_vector_int(desc + "_strides", strides);

        lu << "CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(" << desc << ", " << data_type << ", "
           << static_cast<int>(dimensions.size()) << ", " << desc << "_dim, " << desc << "_strides"
           << "));\n";
    }

    return _lu;
}

LanguageUnit_p
    cuda::get_cudnn_filter_descriptor(const Shape& shape, string desc, element::Type type)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;

    string data_type = cuda::get_cudnn_datatype(type);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnFilterDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&" << desc << "));\n";

    std::vector<int> dimensions(fmax(4, shape.size()), 1);
    int idx = 0;
    for (size_t i = dimensions.size() - shape.size(); i < dimensions.size(); i++)
    {
        dimensions[i] = static_cast<int>(shape[idx++]);
    }

    if (dimensions.size() <= 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(" << desc << ", "
           /*dataType=*/
           << data_type << ", "
           /*format=*/
           << tensor_format << ", "
           /*dimension_size*/
           << dimensions[0] << ", "
           /*dimension_size*/
           << dimensions[1] << ", "
           /*dimension_size*/
           << dimensions[2] << ", "
           /*dimension_size*/
           << dimensions[3] << "));\n";
    }
    else
    {
        lu << "CUDNN_SAFE_CALL("
           << "cudnnSetFilterNdDescriptor(" << desc << ", "
           /*dataType=*/
           << data_type << ", "
           /*format=*/
           << tensor_format << ", "
           /*num_dimensions=*/
           << static_cast<int>(dimensions.size()) << ", "
           /*dimensions*/
           << dimensions.data() << "));\n";
    }
    return _lu;
}

LanguageUnit_p cuda::get_cudnn_convolution_descriptor(const Shape& padding,
                                                      const Strides& window_movement_strides,
                                                      const Strides& window_dilation_strides,
                                                      string desc,
                                                      element::Type type)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;

    string data_type = cuda::get_cudnn_datatype(type);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnConvolutionDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&" << desc << "));\n";

    std::vector<int> window_movement_strides_int(window_movement_strides.size());
    std::vector<int> window_dilation_strides_int(window_dilation_strides.size());
    std::vector<int> padding_int(padding.size());
    for (int i = 0; i < padding.size(); i++)
    {
        window_movement_strides_int[i] = static_cast<int>(window_movement_strides[i]);
        window_dilation_strides_int[i] = static_cast<int>(window_dilation_strides[i]);
        padding_int[i] = static_cast<int>(padding[i]);
    }

    if (padding.size() == 2)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(" << desc << ", " << padding_int[0]
           << ", " << padding_int[1] << ", " << window_movement_strides_int[0] << ", "
           << window_movement_strides_int[1] << ", " << window_dilation_strides_int[0] << ", "
           << window_dilation_strides_int[1] << ", CUDNN_CROSS_CORRELATION, " << data_type
           << "));\n";
    }
    else
    {
        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            NNFUSION_CHECK(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}";
            return ss.str();
        };

        lu << expand_vector_int("padding_int", padding_int) << ";\n";
        lu << expand_vector_int("window_movement_strides_int", window_movement_strides_int)
           << ";\n";
        lu << expand_vector_int("window_dilation_strides_int", window_dilation_strides_int)
           << ";\n";

        lu << "CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(" << desc << ", "
           << static_cast<int>(padding_int.size()) << ", "
           << "padding_int, "
           << "window_movement_strides_int, "
           << "window_dilation_strides_int, CUDNN_CROSS_CORRELATION, " << data_type << "));\n";
    }
    return _lu;
}

LanguageUnit_p cuda::get_dropout_global_states(float ratio)
{
    NNFUSION_CHECK(ratio >= 0 && ratio < 1);
    std::string prefix = "dropout_" + ratio2str(ratio);

    std::string mangled_name = "declaration::" + prefix;
    std::string code = nnfusion::op::create_code_from_template(
        R"(
cudnnDropoutDescriptor_t @prefix@_desc;
void* @prefix@_states;

void @prefix@_init(cudnnHandle_t cudnn_handle)
{
    size_t dropout_state_size;
    CUDNN_SAFE_CALL(cudnnCreateDropoutDescriptor(&@prefix@_desc));
    CUDNN_SAFE_CALL(cudnnDropoutGetStatesSize(cudnn_handle, &dropout_state_size));
    CUDA_SAFE_CALL(cudaMalloc(&@prefix@_states, dropout_state_size));
    CUDNN_SAFE_CALL(cudnnSetDropoutDescriptor(@prefix@_desc, cudnn_handle, @ratio@, @prefix@_states, dropout_state_size, /*seed*/ 0));
}

void @prefix@_free()
{
    CUDNN_SAFE_CALL(cudnnDestroyDropoutDescriptor(@prefix@_desc));
    CUDA_SAFE_CALL(cudaFree(@prefix@_states));
}

)",
        {
            {"prefix", prefix}, {"ratio", ratio},
        });
    LanguageUnit_p lu(new LanguageUnit(mangled_name, code));
    return lu;
}
