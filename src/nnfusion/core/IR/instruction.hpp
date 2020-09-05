// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"

using namespace nnfusion::kernels;

namespace nnfusion
{
    namespace ir
    {
        class Instruction : public Tagable
        {
        public:
            Instruction() {}
            using Pointer = std::shared_ptr<Instruction>;

            Instruction(std::shared_ptr<graph::GNode> gnode);

        private:
            bool has_name_;
            std::string name_;
            bool has_doc_string_;
            std::string doc_string_;

            Attributes _attr;
            std::shared_ptr<graph::GNode> gnode;
            std::shared_ptr<KernelEmitter> kernel;
            // The input tensor ptr
            vector<shared_ptr<nnfusion::descriptor::Tensor>> inputs;
            // The output tensor ptr
            vector<shared_ptr<nnfusion::descriptor::Tensor>> outputs;
            // The tmp tensor ptr
            vector<shared_ptr<nnfusion::descriptor::Tensor>> internal_tensors;
            KernelEmitter::Pointer get_kernel_from_gnode(std::shared_ptr<graph::GNode> gnode);
            void extract_gnode_tensor(std::shared_ptr<graph::GNode> gnode);

        public:
            bool has_name() { return has_name_; }
            const std::string& name() const { return name_; }
            void setName(std::string name)
            {
                has_name_ = true;
                name_ = std::move(name);
            }
            bool has_doc_string() const { return has_doc_string_; }
            const std::string& docString() { return doc_string_; }
            void setDocString(std::string doc_string)
            {
                has_doc_string_ = true;
                doc_string_ = std::move(doc_string);
            }

            void setGNode(std::shared_ptr<graph::GNode> gnode);
            std::shared_ptr<graph::GNode> getGNode() { return gnode; }
            void setKernel(std::shared_ptr<KernelEmitter> kernel);
            std::shared_ptr<KernelEmitter> getKernel() { return kernel; }
            Attributes& Attr() { return _attr; }
            Tags& Tag() { return *this; }
            vector<shared_ptr<nnfusion::descriptor::Tensor>>& get_inputs();
            vector<shared_ptr<nnfusion::descriptor::Tensor>>& get_outputs();
            vector<shared_ptr<nnfusion::descriptor::Tensor>>& get_internal_tensors();
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> liveness_new_list;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> liveness_free_list;
        };
    }
}