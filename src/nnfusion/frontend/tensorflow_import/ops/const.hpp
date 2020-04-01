// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../tensorflow_base.hpp"
#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            NamedNodeVector TranslateConstOp(const tensorflow::NodeDef& node,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_ngraph);
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
