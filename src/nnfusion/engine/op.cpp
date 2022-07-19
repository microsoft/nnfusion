// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "op.hpp"

using namespace nnfusion::ir;

unordered_map<string, LanguageUnit_p> ir::Function::definition_pool;

Operator::Operator()
    : m_name("Null")
    , isTranslated(false)
    , gnode(nullptr)
{
}

Operator::Operator(shared_ptr<graph::GNode> gnode)
    : Operator()
{
    vector<shared_ptr<descriptor::Tensor>> in;
    vector<string> node_input_names;
    vector<string> node_output_names;
    for (size_t i = 0; i < gnode->get_input_size(); i++)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        in.push_back(tv);
        node_input_names.emplace_back(tv->get_name());
    }
    vector<shared_ptr<descriptor::Tensor>> out;
    for (size_t i = 0; i < gnode->get_output_size(); i++)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        out.push_back(tv);
        node_output_names.emplace_back(tv->get_name());
    }

    // Output debug info of node
    if (!gnode->get_op_ptr()->is_tensor_op())
    {
        NNFUSION_LOG(INFO) << "Node:\t" << gnode->get_name() << "\t(";
        vector<string> parameter_nodes = node_input_names;
        parameter_nodes.insert(
            parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
        NNFUSION_LOG(INFO) << join(parameter_nodes);
        NNFUSION_LOG(INFO) << ")\n";
    }

    for (auto arg : in)
    {
        this->dtypes.push_back(arg->get_element_type().c_type_string());
    }

    for (auto ou : out)
    {
        this->dtypes.push_back(ou->get_element_type().c_type_string());
    }

    this->gnode = gnode;
    this->args = in;
    this->arg_names = node_input_names;
    this->out = out;
    this->out_names = node_output_names;
}

ir::Function::Function()
    : definition_unit(nullptr)
    , op(nullptr)
    , call_unit(nullptr)
    , test_unit(nullptr)
    , dep_unit(nullptr)
    , source_unit(nullptr)
    , isCodeGened(false)
{
}

ir::Function::Function(shared_ptr<Operator> op)
    : Function()
{
    NNFUSION_CHECK_NOT_NULLPTR(this->op = op);
}

LanguageUnit_p ir::Function::codegen_source()
{
    NNFUSION_CHECK(isCodeGened == false) << "Code only generated once.";
    NNFUSION_CHECK_NOT_NULLPTR(this->dep_unit = codegen_dependency());
    if (definition_pool.find(codegen_function_name()) != definition_pool.end())
    {
        NNFUSION_CHECK_NOT_NULLPTR(this->definition_unit =
                                       definition_pool[codegen_function_name()]);
    }
    else
    {
        NNFUSION_CHECK_NOT_NULLPTR(this->definition_unit = codegen_function_definition());
    }
    NNFUSION_CHECK_NOT_NULLPTR(this->call_unit = codegen_function_call());
    NNFUSION_CHECK_NOT_NULLPTR(this->test_unit = codegen_test());
    // Pass other to dep_unit
    for (auto& it : call_unit->local_symbol)
        dep_unit->require(it.second);
    for (auto& it : definition_unit->local_symbol)
        dep_unit->require(it.second);
    for (auto& it : test_unit->local_symbol)
        dep_unit->require(it.second);
    call_unit->clean_require();
    definition_unit->clean_require();
    test_unit->clean_require();

    // organize dep
    this->definition_unit->require(this->dep_unit);
    NNFUSION_CHECK(this->call_unit->require(this->definition_unit));
    NNFUSION_CHECK(this->test_unit->require(this->definition_unit));

    isCodeGened = true;
    return this->call_unit;
}