// Microsoft (c) 2019, Wenxiang
#pragma once

#include "cuda_helper.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            std::vector<int> compute_strides(const std::vector<int>& shape);
            std::string get_cudnn_datatype(std::string dtype);
            LanguageUnit_p cudnn_tensor_descriptor_from_shape(const nnfusion::Shape& shape,
                                                              string desc);
            LanguageUnit_p get_cudnn_convolution_descriptor(const Shape& padding,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            string desc);
            LanguageUnit_p get_cudnn_filter_descriptor(const Shape& shape, string desc);
        }
    }
}