// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"
#include "nnfusion/core/graph/gnode.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Instruction : public Tagable
        {
        public:
            Instruction() {}
            using Pointer = std::shared_ptr<Instruction>;

            Instruction(const nnfusion::graph::GNode& gnode);

        private:
            bool has_name_;
            std::string name_;
            bool has_doc_string_;
            std::string doc_string_;

            Attributes _attr;
            std::shared_ptr<graph::GNode> gnode;

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

            void setGNode(std::shared_ptr<graph::GNode> gnode) { this->gnode = gnode; }
            std::shared_ptr<graph::GNode> getGNode() { return gnode; }
            Attributes& Attr() { return _attr; }
            Tags& Tag() { return *this; }
        };
    }
}