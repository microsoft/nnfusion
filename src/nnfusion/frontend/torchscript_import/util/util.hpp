//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#pragma once
#include "../torchscript_base.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            using GNodeVector = graph::GNodeVector;

            GNodePtr
                GetInputNode(const NodeMap& all_ng_nodes, const TNodePtr node, size_t input_idx);

            graph::GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes, const TNodePtr node);

            bool ScalarTypeToNGraphElementType(const c10::ScalarType ts_dt,
                                               nnfusion::element::Type* ng_et);
            bool TypeKindToNGraphElementType(const c10::TypeKind ts_dt,
                                             nnfusion::element::Type* ng_et);

            template <typename T>
            std::vector<T> GetConstValues(GNodePtr n, int64 expect_size = -1)
            {
                NNFUSION_CHECK(n) << "Nullptr found";
                // const nnfusion::element::Type et = nnfusion::element::from<T>();
                std::vector<T> ret;
                NNFUSION_CHECK(GetValueFromNGraphOp<T>(n, &ret)) << "Fail to get value from node: "
                                                                 << n->get_name();
                if (expect_size >= 0)
                {
                    NNFUSION_CHECK(ret.size() == expect_size) << "Expect value size " << expect_size
                                                              << ", but found " << ret.size();
                }
                return ret;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
