// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_emitter.hpp"
#include "nnfusion/engine/async_manager.hpp"

#include <string>

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_bool(fextern_result_memory);
DECLARE_bool(fantares_mode);
DECLARE_string(fdefault_device);

KernelContext::KernelContext(shared_ptr<graph::GNode> gnode)
    : gnode(gnode)
    , op(gnode->get_op_ptr())
    , gpu_num_sm(20)
{
    // extract input tensors
    for (size_t i = 0; i < gnode->get_input_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        inputs.push_back(tv);
        input_names.push_back(tv->get_name());
    }

    // extract output tensors
    for (size_t i = 0; i < gnode->get_output_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        outputs.push_back(tv);
        output_names.push_back(tv->get_name());
    }

    for (auto arg : inputs)
    {
        this->dtypes.push_back(arg->get_element_type().c_type_string());
    }

    for (auto out : outputs)
    {
        this->dtypes.push_back(out->get_element_type().c_type_string());
    }

    annotations = gnode->get_op_ptr()->get_op_annotations();
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx)
    : m_context(ctx)
    , m_is_emitted(false)
    , m_intra_op_parallelism(false)
{
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type)
    : m_context(ctx)
    , m_is_emitted(false)
    , m_kernel_type(kernel_type)
    , m_intra_op_parallelism(false)
{
}

NNFusion_DeviceType KernelEmitter::get_device_type()
{
    if (m_kernel_type == "cuda" || m_kernel_type == "cuda_lib")
    {
        return NNFusion_DeviceType::CUDA_GPU;
    }
    else if (m_kernel_type == "cpu")
    {
        return NNFusion_DeviceType::GENERIC_CPU;
    }
    else if (m_kernel_type == "hlsl")
    {
        return NNFusion_DeviceType::HLSL;
    }
    else if (m_kernel_type == "rocm")
    {
        return NNFusion_DeviceType::ROCM_GPU;
    }
    else if (m_kernel_type == "graphcore")
    {
        return NNFusion_DeviceType::GraphCore;
    }
    return NNFusion_DeviceType::UNKNOWN;
}

LanguageUnit_p KernelEmitter::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit("function_name"));
    auto& lu = *_lu;

    lu << m_context->gnode->get_op_type() << "_" << join(m_context->dtypes, "_") << "_"
       << m_kernel_type << "_"
       << m_context->gnode->get_op_ptr()->get_unique_name(); //<< custom_tag;
    return _lu;
}

LanguageUnit_p KernelEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;
    string::size_type idx = this->m_kernel_name.find("Result");

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        if (idx == string::npos || FLAGS_fextern_result_memory)
        {
            ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        }
        else
        {
            ss << m_context->outputs[i]->get_element_type().c_type_string() << "** ";
        }
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        // default name is: "persit0", "persist1" ...
        ss << m_context->tensors[i]->get_name();
        params.push_back(ss.str());
    }

    lu << "void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p KernelEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    names.insert(names.end(), m_context->tensor_names.begin(), m_context->tensor_names.end());
    lu << "(";

    auto gnode = m_context->gnode;
    if (m_function_unit != nullptr)
    {
        auto sig_unit = m_function_unit->signature_unit;
        NNFUSION_CHECK_NOT_NULLPTR(sig_unit);
        if (gnode && (*gnode)["Async_info"].is_valid())
        {
            auto& async_info = (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
            auto stream = async_info.execution_stream;
            if (stream)
            {
                if (sig_unit->get_code().find("cudaStream_t") != string::npos)
                    lu << stream->get_name() << ", ";

                auto binding_symbol = stream->get_binding_symbol();
                if (sig_unit->get_code().find("cudnnHandle_t") != string::npos)
                {
                    NNFUSION_CHECK(binding_symbol.find("cudnn_handle") != binding_symbol.end());
                    lu << binding_symbol["cudnn_handle"] << ", ";
                }
                if (sig_unit->get_code().find("cublasHandle_t") != string::npos)
                {
                    NNFUSION_CHECK(binding_symbol.find("cublas_handle") != binding_symbol.end());
                    lu << binding_symbol["cublas_handle"] << ", ";
                }
            }
        }
    }
    lu << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p KernelEmitter::emit_comments()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_comments"));
    auto& lu = *_lu;
    lu << "// Node name:\t" << m_context->gnode->get_op_ptr()->get_unique_name() << "\n";
    lu << "// Description:\t" << m_context->gnode->get_op_type() << "\n";
    lu << "// Input:\n";
    for (auto in : m_context->inputs)
    {
        lu << "//\t- name: " << in->get_name();
        lu << "\ttype: " << in->get_element_type().c_type_string();
        lu << "\tshape: " << in->get_shape();
        lu << "\n";
    }

    lu << "// Output:\n";
    for (auto out : m_context->outputs)
    {
        lu << "//\t- name: " << out->get_name();
        lu << "\ttype: " << out->get_element_type().c_type_string();
        lu << "\tshape: " << out->get_shape();
        lu << "\n";
    }

    if (!m_context->tensors.empty())
        lu << "// Other tensors in use:\n";

    for (auto persist : m_context->tensors)
    {
        lu << "//\t- name: " << persist->get_name();
        lu << "\ttype: " << persist->get_element_type().c_type_string();
        lu << "\tshape: " << persist->get_shape();
        lu << "\n";
    }

    return _lu;
}

FunctionUnit_p KernelEmitter::get_or_emit_source(bool emit_func_call)
{
    if (m_is_emitted)
    {
        if (emit_func_call)
            m_function_unit->call_unit = emit_function_call();
        return m_function_unit;
    }
    m_function_unit = emit_source();
    m_is_emitted = true;
    return m_function_unit;
}

FunctionUnit_p KernelEmitter::emit_source()
{
    FunctionUnit_p fu(new FunctionUnit());

    if (this->m_kernel_name.empty())
    {
        fu->name_unit = emit_function_name();
        this->m_kernel_name = fu->name_unit->get_code();
    }

    if (kernel_definitions.find(this->m_kernel_name) != kernel_definitions.end())
    {
        NNFUSION_CHECK_NOT_NULLPTR(fu = kernel_definitions[this->m_kernel_name]);
        return fu;
    }

    // a hack way to map int64 to long long
    auto replace_cstring = [](LanguageUnit_p lp) {
        if (lp && FLAGS_fantares_mode && FLAGS_fdefault_device != "HLSL")
        {
            auto lp_str = lp->get_code();
            // avoid modify naming string
            lp_str = replace_sub_str(lp_str, "_uint64_t", "@underline_unsigned_integer@");
            lp_str = replace_sub_str(lp_str, "_int64_t", "@underline_integer@");
            lp_str = replace_sub_str(lp_str, "uint64_t_", "@unsigned_integer_underline@");
            lp_str = replace_sub_str(lp_str, "int64_t_", "@integer_underline@");

            lp_str = replace_sub_str(lp_str, "uint64_t", "unsigned long long");
            lp_str = replace_sub_str(lp_str, "int64_t", "long long");

            lp_str = replace_sub_str(lp_str, "@underline_unsigned_integer@", "_uint64_t");
            lp_str = replace_sub_str(lp_str, "@underline_integer@", "_int64_t");
            lp_str = replace_sub_str(lp_str, "@unsigned_integer_underline@", "uint64_t_");
            lp_str = replace_sub_str(lp_str, "@integer_underline@", "int64_t_");
            lp->modify_code(lp_str);
        }
        return lp;
    };
    // emit function units
    NNFUSION_CHECK_NOT_NULLPTR(fu->signature_unit = replace_cstring(emit_function_signature()));
    fu->body_unit = replace_cstring(emit_function_body());
    if (!fu->body_unit)
    {
        return nullptr;
    }

    NNFUSION_CHECK_NOT_NULLPTR(fu->call_unit = emit_function_call());
    NNFUSION_CHECK_NOT_NULLPTR(fu->dep_unit = emit_dependency());
    NNFUSION_CHECK_NOT_NULLPTR(fu->comment_unit = emit_comments());

    // Pass other to dep_unit
    for (auto& it : fu->call_unit->local_symbol)
        fu->dep_unit->require(it.second);
    for (auto& it : fu->body_unit->local_symbol)
        fu->dep_unit->require(it.second);
    fu->call_unit->clean_require();
    fu->body_unit->clean_require();

    // organize dep
    NNFUSION_CHECK(fu->body_unit->require(fu->dep_unit));
    NNFUSION_CHECK(fu->call_unit->require(fu->body_unit));

    return fu;
}

const shared_ptr<nnfusion::descriptor::Tensor>
    KernelEmitter::allocate_tensor(Shape shape,
                                   element::Type elt,
                                   string name,
                                   NNFusion_DeviceType device_type,
                                   bool is_persistent,
                                   bool is_constant,
                                   bool is_parameter,
                                   bool is_RDMA_tensor,
                                   const string& group,
                                   int device_id)
{
    // Internal access of this tensor should be like temp0, temp1 ...
    // External access of this tensor should be like Conv1_temp0, Conv2_temp1...
    ///\important Important assumption! the tensor allocated can only be seen inside the kernel.
    string t_name = "temp" + to_string(m_context->tensors.size());
    // Generate tensor name
    if (name.empty())
    {
        name = m_context->gnode->get_op_ptr()->get_unique_name();
        name = name + "_" + t_name;
    }
    auto temp_tensor = make_shared<nnfusion::descriptor::Tensor>(elt,
                                                                 shape,
                                                                 name,
                                                                 device_type,
                                                                 is_persistent,
                                                                 is_constant,
                                                                 is_parameter,
                                                                 is_RDMA_tensor,
                                                                 group,
                                                                 device_id);
    m_context->tensors.push_back(move(temp_tensor));
    m_context->tensor_names.push_back(name);

    NNFUSION_LOG(INFO) << "Tensor allocated:\t" << name << ", shape is:" << shape;
    return m_context->tensors.back();
}

shared_ptr<nnfusion::cache::KernelEntry>
    KernelEmitter::get_kernel_cache_entry(shared_ptr<nnfusion::cache::KernelEntry> kernel_entry)
{
    if (kernel_entry == nullptr)
    {
        kernel_entry = std::make_shared<nnfusion::cache::KernelEntry>();
    }
    FunctionUnit_p func_p = this->get_or_emit_source();
    if (func_p == nullptr)
    {
        NNFUSION_LOG(ERROR) << "Cannot generate kernel_cache_entry due to invalid KernelEmitter: "
                            << m_context->gnode->get_name();
        return nullptr;
    }

    if (kernel_entry->key == "")
    {
        kernel_entry->key = func_p->body_unit->get_code();
    }

    if (kernel_entry->identifier == "")
    {
        kernel_entry->identifier = m_context->generate_identifier();
    }

    if (kernel_entry->op_type == "")
    {
        kernel_entry->op_type = m_context->gnode->get_op_type();
    }

    if (kernel_entry->attributes.is_null())
    {
        kernel_entry->attributes = nlohmann::json();
    }

    if (kernel_entry->source == "")
    {
        kernel_entry->source = "NNFusion";
    }

    if (kernel_entry->device_type == "")
    {
        kernel_entry->device_type = get_device_str(this->get_device_type());
    }

    if (kernel_entry->function.is_null())
    {
        kernel_entry->function = nlohmann::json();
    }
    if (kernel_entry->function.find("function_signature") == kernel_entry->function.end())
    {
        kernel_entry->function["function_signature"] = func_p->signature_unit->get_code();
    }
    if (kernel_entry->function.find("function_body") == kernel_entry->function.end())
    {
        kernel_entry->function["function_body"] = func_p->body_unit->get_code();
    }
    if (kernel_entry->function.find("function_dep") == kernel_entry->function.end())
    {
        kernel_entry->function["function_dep"] = func_p->dep_unit->get_code();
    }

    if (kernel_entry->tags.size() == 0)
    {
        kernel_entry->tags = std::set<std::string>();
    }
    kernel_entry->tags.insert("KernelEmitter");

    if (kernel_entry->miscs.is_null())
    {
        kernel_entry->miscs = nlohmann::json();
    }

    bool is_valid_entry = kernel_entry->key != "" && kernel_entry->identifier != "" &&
                          kernel_entry->op_type != "" && kernel_entry->source != "" &&
                          kernel_entry->device_type != "" && kernel_entry->function.dump() != "";

    return is_valid_entry ? kernel_entry : nullptr;
}

template <typename Iter>
std::string item_join(Iter begin,
                      Iter end,
                      std::string const& separator,
                      std::function<std::string(Iter)> f_item)
{
    std::ostringstream result;
    if (begin != end)
        result << f_item(begin++);
    while (begin != end)
        result << separator << f_item(begin++);
    return result.str();
}

std::string nnfusion::kernels::KernelContext::generate_identifier()
{
    auto ctx = this;

    std::string op_type = ctx->gnode->get_op_type();

    // identifier of pattern substitution kernel was generated before
    if (op_type == "Matched_Pattern")
    {
        if ((*ctx->gnode)["identifier"].is_valid())
        {
            return (*ctx->gnode)["identifier"].as<std::string>();
        }
        else
        {
            return "";
        }
    }

    // Todo: more spec to be added
    std::string identifier("");

    // operator type as identifier
    identifier += op_type;

    // shapes of input and output tensors as identifier
    std::function<std::string(std::vector<size_t>::const_iterator)> f_shape =
        [](std::vector<size_t>::const_iterator s) { return to_string(*s); };
    std::function<std::string(std::vector<std::shared_ptr<nnfusion::descriptor::Tensor>>::iterator)>
        f_tensor =
            [&f_shape](std::vector<std::shared_ptr<nnfusion::descriptor::Tensor>>::iterator t) {
                auto& shape = (*t)->get_shape();
                return item_join(shape.begin(), shape.end(), ",", f_shape);
            };
    identifier += item_join(ctx->inputs.begin(), ctx->inputs.end(), ";", f_tensor);
    identifier += ";" + item_join(ctx->outputs.begin(), ctx->outputs.end(), ";", f_tensor);

    // data types of input tensors as identifier
    for (int i = 0; i < ctx->dtypes.size(); ++i)
    {
        identifier += ctx->dtypes[i];
    }

    if (op_type == "Convolution")
    {
        auto conv = std::dynamic_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(conv);
        std::stringstream str;
        str << conv->get_window_movement_strides();
        str << conv->get_window_dilation_strides();
        str << conv->get_padding_below();
        identifier += str.str();
    }
    else if (op_type == "AvgPool")
    {
        auto avgpool = std::dynamic_pointer_cast<op::AvgPool>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(avgpool);
        std::stringstream str;
        str << avgpool->get_window_shape();
        str << avgpool->get_window_movement_strides();
        str << avgpool->get_padding_below();
        str << avgpool->get_padding_above();
        identifier += str.str();
    }
    else if (op_type == "MaxPool")
    {
        auto maxpool = std::dynamic_pointer_cast<op::MaxPool>(ctx->gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(maxpool);
        std::stringstream str;
        str << maxpool->get_window_shape();
        str << maxpool->get_window_movement_strides();
        str << maxpool->get_padding_below();
        str << maxpool->get_padding_above();
        identifier += str.str();
    }
    else if (op_type == "Dot")
    {
        ///\todo encode dot attrs, stay the same with db importor
        // auto dot = std::dynamic_pointer_cast<op::Dot>(ctx->gnode->get_op_ptr());
        // NNFUSION_CHECK_NOT_NULLPTR(dot);
        // std::stringstream str;
        // str << dot->get_transpose_A();
        // str << dot->get_transpose_B();
        // ///\todo: need to encode dot reduction_axes_count?
        // identifier += str.str();
    }

    return identifier;
}