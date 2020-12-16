// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iomanip>
#include "cuda_helper.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            std::vector<int> compute_strides(const std::vector<int>& shape);
            std::string get_cudnn_datatype(element::Type type);
            LanguageUnit_p cudnn_tensor_descriptor_from_shape(const nnfusion::Shape& shape,
                                                              string desc,
                                                              element::Type type);
            LanguageUnit_p get_cudnn_convolution_descriptor(const Shape& padding,
                                                            const Strides& window_movement_strides,
                                                            const Strides& window_dilation_strides,
                                                            string desc,
                                                            element::Type type = element::f32);
            LanguageUnit_p get_cudnn_filter_descriptor(const Shape& shape,
                                                       string desc,
                                                       element::Type type = element::f32);
            LanguageUnit_p get_dropout_global_states(float ratio);
            inline std::string ratio2str(float ratio)
            {
                // convert ratio to a legal c-style variable name
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << ratio;
                std::string s = stream.str();
                ///\todo handle 1e-6
                auto found = s.find(".");
                if (found != std::string::npos)
                    s = s.replace(found, 1, "_");
                return s;
            }
        }
    }
}