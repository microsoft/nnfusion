//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT License.

#include <unordered_map>

#include "parameter.hpp"

namespace nnfusion
{
    namespace frontend
    {
        static const std::unordered_map<std::string, const nnfusion::element::Type>
            string2ElementType{{"float", nnfusion::element::f32},
                               {"double", nnfusion::element::f64},
                               {"int8_t", nnfusion::element::i8},
                               {"int16_t", nnfusion::element::i16},
                               {"int32_t", nnfusion::element::i32},
                               {"int64_t", nnfusion::element::i64},
                               {"uint8_t", nnfusion::element::u8},
                               {"uint16_t", nnfusion::element::u16},
                               {"uint32_t", nnfusion::element::u32},
                               {"uint64_t", nnfusion::element::u64}};
        namespace
        {
            std::vector<std::string> split_string(const std::string& s,
                                                  const std::string& delimiter)
            {
                size_t last = 0;
                size_t next = 0;
                std::vector<std::string> result;
                while ((next = s.find(delimiter, last)) != std::string::npos)
                {
                    result.push_back(s.substr(last, next - last));
                    last = next + 1;
                }
                result.push_back(s.substr(last, next - last));
                return result;
            }
        }

        // ParamInfo::ParamInfo(const nnfusion::Shape& shape, nnfusion::element::Type type): shape(shape), type(type) {}
        ParamInfo::ParamInfo(const nnfusion::Shape& shape, nnfusion::element::Type type)
            : shape(shape)
            , type(type)
        {
        }

        ParamInfo::ParamInfo(const nnfusion::Shape& shape, const std::string& type_s)
            : shape(shape)
        {
            auto it = string2ElementType.end();
            if (!type_s.empty())
            {
                it = string2ElementType.find(type_s);
                NNFUSION_CHECK(it != string2ElementType.end()) << "Unsupported type: " << type_s;
            }
            else
            {
                it = string2ElementType.find("float");
            }
            this->type = type;
        }

        ParamInfo::ParamInfo(const std::string& s_full)
        {
            auto shape_and_type = split_string(s_full, ":");
            NNFUSION_CHECK(shape_and_type.size() == 2);
            auto shape_s = split_string(shape_and_type[0], ",");

            auto type_s = shape_and_type[1];

            for (auto s : shape_s)
            {
                this->shape.push_back(std::stoull(s));
            }

            auto it = string2ElementType.end();
            if (!type_s.empty())
            {
                it = string2ElementType.find(type_s);
                NNFUSION_CHECK(it != string2ElementType.end()) << "Unsupported type: " << type_s;
            }
            else
            {
                it = string2ElementType.find("float");
            }
            this->type = it->second;
        }

        std::vector<ParamInfo> build_torchscript_params_from_string(const std::string& ss)
        {
            std::vector<ParamInfo> params;
            for (auto s : split_string(ss, ";"))
            {
                params.emplace_back(s);
            }
            return params;
        }

        std::unordered_map<std::string, size_t> build_onnx_params_from_string(const std::string& ss)
        {
            std::unordered_map<std::string, size_t> ret;
            for (auto s : split_string(ss, ";"))
            {
                auto dim_info = split_string(s, ":");
                NNFUSION_CHECK(dim_info.size() == 2) << "illegal dim info " << s;
                ret.emplace(dim_info[0], std::stoull(dim_info[1]));
            }
            return ret;
        }
    } // namespace frontend
} // namespace nnfusion