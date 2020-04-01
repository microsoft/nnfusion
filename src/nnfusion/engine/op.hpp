// Microsoft (c) 2019, Wenxiang
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace ir
    {
        // Store the caculated intermediated data
        class Operator
        {
        public:
            // Common Data
            string m_name;
            bool isTranslated;

        public:
            shared_ptr<graph::GNode> gnode;
            vector<shared_ptr<descriptor::Tensor>> args;
            vector<string> arg_names;
            vector<shared_ptr<descriptor::Tensor>> out;
            vector<string> out_names;
            vector<string> dtypes;

            Operator();
            Operator(shared_ptr<graph::GNode> gnode);
            ~Operator(){};
        };

        using Operator_p = shared_ptr<Operator>;

        class Group : public Operator
        {
        private:
            // set<IntermediateOP> operators;
        };

        using Group_p = shared_ptr<Group>;

        // Generate Solution files
        class Function : public Operator
        {
        protected:
            bool isCodeGened;
            // mapping: kernel name -> kernel definition
            static unordered_map<string, LanguageUnit_p> definition_pool;

        public:
            // Common Data
            Operator_p op;
            LanguageUnit_p definition_unit;
            LanguageUnit_p call_unit;
            LanguageUnit_p source_unit;
            LanguageUnit_p dep_unit;
            LanguageUnit_p test_unit;
            LanguageUnit_p test_call_unit;

            // Get the property of some CodeGenOP
            virtual string codegen_function_name() = 0;
            virtual string codegen_test_name() = 0;

            // Interface for Generate code pieces
            virtual LanguageUnit_p codegen_dependency() = 0;
            virtual LanguageUnit_p codegen_function_definition() = 0;
            virtual LanguageUnit_p codegen_function_call() = 0;
            virtual LanguageUnit_p codegen_test() = 0;
            virtual LanguageUnit_p codegen_source();

            bool is_codegened() { return isCodeGened; }
            Function();
            Function(Operator_p inter_op);
        };

        using Function_p = shared_ptr<Function>;

        // Through JIT(NVRTC)
        class Module : public Function
        {
        protected:
            string source;
            bool isCompiled;

        public:
            virtual shared_ptr<Module> compile() = 0;
        };
    }
}