// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>

#include "../tensorflow_base.hpp"
#include "../util/util.hpp"
#include "nnfusion/common/type/element_type.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            NamedNodeVector TranslateConstOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_ngraph);

            // extern const std::map<tensorflow::DataType, element::Type> TF_NGRAPH_CONST_MAP;
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
