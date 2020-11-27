// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iomanip>
#include <limits>
#include "nnfusion/common/common.hpp"

#define REGISTER_OP(op_x)                                                                          \
    static nnfusion::op::OpConfig __register_op_##op_x = nnfusion::op::build_op_config(#op_x)
#define GENERIC_OP_LOGGING()                                                                       \
    NNFUSION_LOG(DEBUG) << "[GENERIC_OP_LOGGING] " << __FILE__ << ": " << __PRETTY_FUNCTION__;

namespace nnfusion
{
    namespace op
    {
        class OpConfig;
        class GenericOp;

        class OpConfig
        {
        public:
            using any = nlohmann::json;
            using constrait_func_t = bool (*)(const OpConfig::any& config);
            using infershape_func_t = void (*)(std::shared_ptr<graph::GNode> gnode);
            using infersharedmemory_func_t = void (*)(std::shared_ptr<graph::GNode> gnode);
            using translate_func_t = std::string (*)(std::shared_ptr<graph::GNode> gnode);
            using translate_func_t_v2 = std::string (*)(std::shared_ptr<graph::GNode> gnode);

            // OpConfig(): f_infershape(infershape::copy_shape_from_inputs) { }

            template <typename T>
            OpConfig& attr(const std::string& name, const T& val = T())
            {
                getRoot()[name] = val;
                return *this;
            }

            OpConfig& check_constrait()
            {
                NNFUSION_CHECK(is_legal()) << "OpConfig::check_constrait() not passed!";
                return *this;
            }

            OpConfig& constrait(const constrait_func_t& func)
            {
                f_constraits.push_back(func);
                return *this;
            }

            OpConfig& infershape(const infershape_func_t& func)
            {
                f_infershape = func;
                return *this;
            }

            OpConfig& infersharedmemory(const infersharedmemory_func_t& func)
            {
                f_infersharedmemory = func;
                return *this;
            }

            OpConfig& translate(const translate_func_t& func)
            {
                f_translate = func;
                return *this;
            }

            OpConfig& translate_v2(const translate_func_t_v2& func)
            {
                f_translate_v2 = func;
                return *this;
            }

            OpConfig& show()
            {
                NNFUSION_LOG(INFO) << getRoot();
                return *this;
            }

            bool is_legal()
            {
                if (!f_infershape)
                    return false;
                for (auto& func : f_constraits)
                    if (!func(getRoot()))
                        return false;
                return true;
            }

            OpConfig::any& getRoot() { return this->j_attrs["config"]; }
            OpConfig::any& get(std::string key) { return getRoot()[key]; }
            std::vector<constrait_func_t> f_constraits;
            infershape_func_t f_infershape;
            infersharedmemory_func_t f_infersharedmemory;
            translate_func_t f_translate;
            translate_func_t_v2 f_translate_v2;
            OpConfig::any j_attrs;
        };

        std::unordered_map<std::string, OpConfig>& get_op_configs();
        std::string get_translation(std::shared_ptr<nnfusion::graph::GNode>& gnode);
        std::string get_translation_v2(std::shared_ptr<nnfusion::graph::GNode>& gnode);
        std::string get_annotation(std::string translation);

        inline const OpConfig& lookup_op_config(const std::string& opname)
        {
            auto it = get_op_configs().find(opname);
            NNFUSION_CHECK(it != get_op_configs().end())
                << "No config-definition found for op type `" + opname + "`";
            return it->second;
        }

        inline OpConfig& build_op_config(const std::string& opname)
        {
            NNFUSION_CHECK(get_op_configs().find(opname) == get_op_configs().end())
                << "OpConfig for opname `" + opname + "` is registered more than once.";
            //NNFUSION_LOG(INFO) << "Registering opname `" << opname << "`";
            return get_op_configs()[opname];
        }

        template <typename T>
        std::string expand_vector(string name, vector<T>& d, std::string typestring)
        {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << typestring << " " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        }

        inline std::string create_code_from_template(std::string templ,
                                                     const OpConfig::any& feed_dict)
        {
            std::unordered_map<std::string, std::string> feed_pairs;
            for (auto& it : feed_dict.items())
            {
                std::string value;
                if (it.value().is_string())
                    value = it.value();
                else if (it.value().is_null())
                    value = "NULL";
                else if (it.value().is_number_float())
                {
                    std::stringstream ss;
                    ss.flags(std::ios_base::scientific);
                    ss << std::setprecision(std::numeric_limits<float>::digits)
                       << (float)it.value();
                    value = ss.str();
                }
                else
                {
                    std::stringstream ss;
                    ss << it.value();
                    value = ss.str();
                }
                feed_pairs.insert(std::make_pair("@" + it.key() + "@", value));
            }

            int at = templ.find("@");
            while (true)
            {
                int cur_at = templ.find("@", at + 1);
                if (cur_at == std::string::npos || at == std::string::npos)
                    break;
                auto placeholder = templ.substr(at, cur_at - at + 1);
                // NNFUSION_LOG(INFO) << placeholder;
                auto feed_pair = feed_pairs.find(placeholder);
                if (feed_pair != feed_pairs.end())
                {
                    // NNFUSION_LOG(INFO) << "Value->" << feed_pair->second;
                    templ = templ.substr(0, at) + feed_pair->second + templ.substr(cur_at + 1);
                    at += feed_pair->second.size();
                }
                else
                {
                    at = cur_at;
                }
                // NNFUSION_LOG(INFO) << templ;
            }
            return std::move(templ);
        };

        inline std::vector<std::string> create_layout_from_dims(nnfusion::Shape shape)
        {
            std::vector<std::string> shape_def;
            for (int d = 0; d < shape.size(); d++)
            {
                shape_def.push_back(shape[d] == 0 ? "1" : ("N" + to_string(d)));
            }
            return shape_def;
        }

        inline void create_inputs_definition_from_tensor(
            std::shared_ptr<nnfusion::descriptor::Tensor> tensor,
            std::string input_name,
            OpConfig::any& config,
            std::string alias_name = "")
        {
            alias_name = alias_name.empty() ? input_name : alias_name;
            config[alias_name] = input_name;
            auto d_type = tensor->get_element_type().c_type_string();
            if (d_type == "float")
            {
                config[alias_name + "_dtype"] = "float32";
            }
            else if (d_type == "int32_t")
            {
                config[alias_name + "_dtype"] = "int32";
            }
            else if (d_type == "int64_t")
            {
                config[alias_name + "_dtype"] = "int64";
            }
            else
            {
                printf("Unhandled type: %s\n", d_type.c_str());
                assert(0);
            }
            auto shape = tensor->get_shape();
            if (shape.size() == 0)
                shape = {1};
            config[alias_name + "_shape"] = vector_to_string(shape);
        }

        class GenericOp : public Op
        {
        public:
            GenericOp(const std::string& name,
                      const std::string& opname,
                      const OpConfig::any& customOpConfig)
                : Op(opname)
            {
                // Merge customOpConfig into default config
                localOpConfig = lookup_op_config(opname);
                std::unordered_set<std::string> keyset;
                for (auto& item : localOpConfig.getRoot().items())
                    keyset.insert(item.key());

                for (auto& item : customOpConfig.items())
                {
                    NNFUSION_CHECK(keyset.find(item.key()) != keyset.end())
                        << "####### From " << opname << "'s config: " << customOpConfig
                        << ": Invalid attribution `" + item.key() +
                               "` not recognized by op type `" + opname + "`";
                    localOpConfig.getRoot()[item.key()] = item.value();
                }

                set_name(name);
                // NNFUSION_LOG(INFO) << "Managing GenericOp for Opeartor: type = " << opname
                //                    << ", name = " << name;

                localOpConfig.check_constrait();
            }

            virtual void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override
            {
                localOpConfig.check_constrait();
                localOpConfig.f_infershape(gnode);

                if (localOpConfig.f_translate_v2 != nullptr && !m_expression.size())
                {
                    m_expression = localOpConfig.f_translate_v2(gnode);
                }

                if (localOpConfig.f_translate != nullptr && !m_expression.size())
                {
                    m_expression = localOpConfig.f_translate(gnode);
                }
            }

            virtual void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override
            {
                if (localOpConfig.f_infersharedmemory)
                    localOpConfig.f_infersharedmemory(gnode);
            }

            mutable OpConfig localOpConfig;
            std::string m_expression;
        };

        namespace infershape
        {
            // Provide default infershape function: output_shapes[*] = input_shapes[*];
            inline void copy_shape_from_inputs(std::shared_ptr<graph::GNode> gnode)
            {
                for (int i = 0; i < gnode->get_input_size(); ++i)
                {
                    gnode->set_output_type_and_shape(
                        0, gnode->get_input_element_type(i), gnode->get_input_shape(i));
                }
            }

            // unimplemented that will notify exception when migrating to op_v2 mode
            inline void unimplemented_and_not_used(std::shared_ptr<graph::GNode> gnode)
            {
                throw std::runtime_error(
                    ("Not implemented infershape for Op: " + gnode->get_op_ptr()->get_op_type())
                        .c_str());
            }

            inline void copy_shape_from_inputs_with_boolean(std::shared_ptr<graph::GNode> gnode)
            {
                for (int i = 0; i < gnode->get_input_size(); ++i)
                {
                    gnode->set_output_type_and_shape(
                        0, nnfusion::element::boolean, gnode->get_input_shape(i));
                }
            }
        } // namespace infershape
    }     // namespace op
} // namespace nnfusion
