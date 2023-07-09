//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include "util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace torchscript_import
        {
            bool ScalarTypeToNGraphElementType(const c10::ScalarType ts_dt,
                                               nnfusion::element::Type* ng_et)
            {
                switch (ts_dt)
                {
                case c10::ScalarType::Byte: *ng_et = nnfusion::element::u8; break;
                case c10::ScalarType::Char: *ng_et = nnfusion::element::i8; break;
                case c10::ScalarType::Short: *ng_et = nnfusion::element::i16; break;
                case c10::ScalarType::Int: *ng_et = nnfusion::element::i32; break;
                case c10::ScalarType::Long: *ng_et = nnfusion::element::i64; break;
                case c10::ScalarType::BFloat16: *ng_et = nnfusion::element::bf16; break;
                case c10::ScalarType::Float: *ng_et = nnfusion::element::f32; break;
                case c10::ScalarType::Double:
                    *ng_et = nnfusion::element::f32;
                    break; // workaround, convert double to f32
                case c10::ScalarType::Bool: *ng_et = nnfusion::element::boolean; break;
                default: return false;
                }
                return true;
            }

            bool TypeKindToNGraphElementType(const c10::TypeKind ts_dt,
                                             nnfusion::element::Type* ng_et)
            {
                switch (ts_dt)
                {
                case c10::TypeKind::BoolType: *ng_et = nnfusion::element::boolean; break;
                case c10::TypeKind::StringType: *ng_et = nnfusion::element::character; break;
                case c10::TypeKind::IntType: *ng_et = nnfusion::element::i64; break;
                case c10::TypeKind::FloatType:
                    *ng_et = nnfusion::element::f32;
                    break; // workaround, convert double to f32
                default: return false;
                }
                return true;
            }

            GNodePtr
                GetInputNode(const NodeMap& all_ng_nodes, const TNodePtr node, size_t input_idx)
            {
                GNodeVector input_gnodes;
                auto v = node->input(input_idx);
                return all_ng_nodes.at(v->node()).at(v->offset());
            }

            GNodeVector GetAllInputNode(const NodeMap& all_ng_nodes, const TNodePtr node)
            {
                GNodeVector nodes;
                for (size_t i = 0; i < node->inputs().size(); i++)
                {
                    nodes.push_back(GetInputNode(all_ng_nodes, node, i));
                }
                return nodes;
            }
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace nnfusion
