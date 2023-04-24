#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateRecursionOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph);

                NamedNodeVector TranslateRecursionOp(
                    const onnx::NodeProto& node_proto,
                    const onnx::GraphProto& graph_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    const std::unordered_map<std::string, ConvertFuncMap>& domain_convert_func_map,
                    const string& model_dir,
                    const std::unordered_map<std::string, std::int64_t>& domain2version,
                    const std::unordered_map<std::string, size_t>& dim_params = {});

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
