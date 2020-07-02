// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ngraph/assertion.hpp"
#include "ngraph/except.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            struct NotSupported : AssertionFailure
            {
                explicit NotSupported(const std::string& what_arg)
                    : AssertionFailure(what_arg)
                {
                }
            };

            struct InvalidArgument : AssertionFailure
            {
                explicit InvalidArgument(const std::string& what_arg)
                    : AssertionFailure(what_arg)
                {
                }
            };

        } // namespace  error

    } // namespace  onnx_import

} // namespace  ngraph

#define ASSERT_IS_SUPPORTED(node_, cond_)                                                          \
    NGRAPH_ASSERT_STREAM(ngraph::onnx_import::error::NotSupported, cond_) << (node_) << " "
#define ASSERT_VALID_ARGUMENT(node_, cond_)                                                        \
    NGRAPH_ASSERT_STREAM(ngraph::onnx_import::error::InvalidArgument, cond_) << (node_) << " "
