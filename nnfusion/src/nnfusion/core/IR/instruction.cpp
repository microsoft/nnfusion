// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#include "instruction.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::ir;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;

Instruction::Instruction(std::shared_ptr<graph::GNode> gnode)
    : gnode(gnode)
    , kernel(get_kernel_from_gnode(gnode))
{
    this->copy_tags_from(*gnode);
    if (!this->kernel && this->gnode)
        this->extract_gnode_tensor(this->gnode);
}

KernelEmitter::Pointer Instruction::get_kernel_from_gnode(std::shared_ptr<graph::GNode> gnode)
{
    if (!(*gnode)["Kernel_Selection_Result"].is_valid())
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be selected before translating:"
                                       << gnode->get_name();
        return nullptr;
    }
    KernelEmitter::Pointer kernel = nullptr;
    auto emitted_kernel =
        (*gnode)["Kernel_Selection_Result"].as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();

    // constant kernel emitter will write file to save weights, skip to do it when codegen.
    if (!gnode->is_constant() && emitted_kernel.second->get_or_emit_source() == nullptr)
        NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before translating:"
                                       << gnode->get_name();
    kernel = emitted_kernel.second;
    return kernel;
}

void Instruction::setGNode(std::shared_ptr<graph::GNode> gnode)
{
    NNFUSION_CHECK(this->gnode == nullptr) << "Instruction's gnode should be set only once.";
    this->gnode = gnode;
    NNFUSION_CHECK(this->kernel == nullptr) << "Instruction's kernel should be set only once.";
    this->kernel = this->get_kernel_from_gnode(gnode);
    if (!this->kernel && this->gnode)
        this->extract_gnode_tensor(this->gnode);
}

void Instruction::setKernel(std::shared_ptr<KernelEmitter> kernel)
{
    NNFUSION_CHECK(this->kernel == nullptr) << "Instruction's kernel should be set only once.";
    this->kernel = kernel;
}

vector<shared_ptr<nnfusion::descriptor::Tensor>>& Instruction::get_inputs()
{
    if (this->kernel)
        return kernel->m_context->inputs;
    return this->inputs;
}
vector<shared_ptr<nnfusion::descriptor::Tensor>>& Instruction::get_outputs()
{
    if (this->kernel)
        return kernel->m_context->outputs;
    return this->outputs;
}

vector<shared_ptr<nnfusion::descriptor::Tensor>>& Instruction::get_internal_tensors()
{
    if (this->kernel)
        return kernel->m_context->tensors;
    return this->internal_tensors;
}

void Instruction::extract_gnode_tensor(std::shared_ptr<graph::GNode> gnode)
{
    // extract input tensors
    for (size_t i = 0; i < gnode->get_input_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        inputs.push_back(tv);
    }

    // extract output tensors
    for (size_t i = 0; i < gnode->get_output_size(); ++i)
    {
        shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
        NNFUSION_CHECK_NOT_NULLPTR(tv);
        outputs.push_back(tv);
    }
}