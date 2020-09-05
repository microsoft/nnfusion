//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once

#include "../torchscript_base.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            std::shared_ptr<op::Op> MakeConstOp(at::Tensor);

            template <typename ShapeType, typename VecType>
            std::shared_ptr<op::Op> MakeConstOp(ShapeType shape, const std::vector<VecType>& values)
            {
                const nnfusion::element::Type et = nnfusion::element::from<VecType>();
                const nnfusion::Shape es = nnfusion::Shape(shape.begin(), shape.end());

                auto ret = std::make_shared<op::Constant>(et, es, values);
                return ret;
            }

            template <typename VecType>
            std::shared_ptr<op::Op> MakeConstOp(std::initializer_list<size_t> shape,
                                                const std::vector<VecType>& values)
            {
                const nnfusion::element::Type et = nnfusion::element::from<VecType>();
                const nnfusion::Shape es = nnfusion::Shape(shape.begin(), shape.end());

                auto ret = std::make_shared<op::Constant>(et, es, values);
                return ret;
            }

            std::shared_ptr<op::Op> MakeConstOp(TNodePtr);

            bool isNoneConst(const std::shared_ptr<op::Op>);
            std::shared_ptr<op::Op> createNoneConst();

            bool constEqual(const GNodePtr, const GNodePtr);
        } // namespace torchscript_import
    }     // namespace frontend
} // namespace nnfusion