// Microsoft (c) 2019, NNFusion Team

#include "kernel_emitter.hpp"
#include <string>

using namespace nnfusion;
using namespace nnfusion::kernels;

KernelContext::KernelContext(shared_ptr<graph::GNode> gnode)
    : gnode(gnode)
    , gpu_num_sm(20)
{
    // extract input tensors
    for (size_t i = 0; i < gnode->get_input_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
        CHECK_NOT_NULLPTR(tv);
        inputs.push_back(tv);
        input_names.push_back(tv->get_name());
    }

    // extract output tensors
    for (size_t i = 0; i < gnode->get_output_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
        CHECK_NOT_NULLPTR(tv);
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
{
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type)
    : m_context(ctx)
    , m_is_emitted(false)
    , m_kernel_type(kernel_type)
{
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
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->tensors.size(); i++)
    {
        stringstream ss;
        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
        // defult name is: "persit0", "persist1" ...
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
    lu << "(" << join(names, ", ") << ");\n";
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

FunctionUnit_p KernelEmitter::get_or_emit_source()
{
    if (m_is_emitted)
    {
        m_function_unit->call_unit = emit_function_call();
        return m_function_unit;
    }

    FunctionUnit_p fu(new FunctionUnit());

    if (this->m_kernel_name.empty())
    {
        fu->name_unit = emit_function_name();
        this->m_kernel_name = fu->name_unit->get_code();
    }

    if (kernel_definitions.find(this->m_kernel_name) != kernel_definitions.end())
    {
        CHECK_NOT_NULLPTR(fu = kernel_definitions[this->m_kernel_name]);
        return fu;
    }

    // emit function units
    CHECK_NOT_NULLPTR(fu->signature_unit = emit_function_signature());
    fu->body_unit = emit_function_body();
    if (!fu->body_unit)
    {
        return nullptr;
    }

    CHECK_NOT_NULLPTR(fu->call_unit = emit_function_call());
    CHECK_NOT_NULLPTR(fu->dep_unit = emit_dependency());
    CHECK_NOT_NULLPTR(fu->comment_unit = emit_comments());

    // Pass other to dep_unit
    for (auto& it : fu->call_unit->local_symbol)
        fu->dep_unit->require(it.second);
    for (auto& it : fu->body_unit->local_symbol)
        fu->dep_unit->require(it.second);
    fu->call_unit->clean_require();
    fu->body_unit->clean_require();

    // orgnize dep
    CHECK(fu->body_unit->require(fu->dep_unit));
    CHECK(fu->call_unit->require(fu->body_unit));
    m_function_unit = fu;
    m_is_emitted = true;
    return fu;
}

const shared_ptr<nnfusion::descriptor::Tensor> KernelEmitter::allocate_tensor(Shape shape,
                                                                              element::Type elt,
                                                                              string name,
                                                                              bool is_persistent,
                                                                              bool is_constant,
                                                                              bool is_parameter,
                                                                              bool is_RDMA_tensor,
                                                                              size_t group_id,
                                                                              size_t device_id)
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
                                                                 is_persistent,
                                                                 is_constant,
                                                                 is_parameter,
                                                                 is_RDMA_tensor,
                                                                 group_id,
                                                                 device_id);
    m_context->tensors.push_back(move(temp_tensor));
    m_context->tensor_names.push_back(name);

    LOG(INFO) << "Tensor allocated:\t" << name << ", shape is:" << shape;
    return m_context->tensors.back();
}

const shared_ptr<nnfusion::descriptor::Tensor>
    KernelEmitter::allocate_tensor(Shape shape,
                                   DeviceType device_type,
                                   element::Type elt,
                                   string name,
                                   bool is_persistent,
                                   bool is_constant,
                                   bool is_parameter,
                                   bool is_RDMA_tensor,
                                   size_t group_id,
                                   size_t device_id)
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
                                                                 group_id,
                                                                 device_id);
    m_context->tensors.push_back(move(temp_tensor));
    m_context->tensor_names.push_back(name);

    LOG(INFO) << "Tensor allocated:\t" << name << ", shape is:" << shape;
    return m_context->tensors.back();
}