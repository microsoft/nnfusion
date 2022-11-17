// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common.hpp"
using namespace std;

namespace nnfusion
{
    /* Support type of language unit
        1. Function Declaration:  void fun();
        2. Function Definition: void fun(){}
        3. Variable: int x = 10;
        4. Import Block: #include <cuda.h>
        5. Function Call
        6. Flag: #define

        lanuage unit: composited of ... vector<lanuage unit>
        depends on: vector<lanuage unit>

        scope: Global Local
        visibility: Global
    */
    class LanguageUnit : public nnfusion::codegen::CodeWriter
    {
    public:
        // Indicate the symbol of current Language Unit
        string symbol;
        // Indicate the required symbols of current LU
        unordered_set<string> required;
        // The renaming map for symbols
        // shared_ptr<unordered_map<string, string>> rename_map;
        unordered_map<string, shared_ptr<LanguageUnit>> local_symbol;
        std::string pwd;
        std::string write_to;
        std::string read_from;
        std::vector<std::pair<std::string, std::string>> copy_templates;
        std::vector<std::string> copy_folder;
        std::string header_code, source_code;
        shared_ptr<LanguageUnit> extern_decl_unit;

    public:
        LanguageUnit()
            : CodeWriter(){};
        LanguageUnit(const string symbol);
        LanguageUnit(const string symbol, const string code);
        LanguageUnit(const string symbol,
                     const string code,
                     const string header,
                     const string source);

        bool change_symbol(const string symbol);
        bool require(const string required);
        void clean_require();
        bool require(shared_ptr<LanguageUnit> lu);
        bool remove(shared_ptr<LanguageUnit> lu);
        bool replace(shared_ptr<LanguageUnit> a, shared_ptr<LanguageUnit> b);
        string get_symbol() { return symbol; }
        string collect_code();
        string collect_required_code();
        virtual void execute(bool append = true);
        virtual void divide_code();

    protected:
        virtual bool setpwd();
        virtual bool write_to_file(bool append);
        virtual bool read_from_file();
        virtual bool copy_all();
    };

    using LanguageUnit_p = shared_ptr<LanguageUnit>;

    struct FunctionUnit : public LanguageUnit
    {
        // Language units to represent a function
        LanguageUnit_p name_unit;
        LanguageUnit_p signature_unit; // void (float* input0, float* input1, float* output0)
        LanguageUnit_p body_unit;
        LanguageUnit_p dep_unit;
        LanguageUnit_p call_unit; // (tensor1, tensor2, tensor3)
        LanguageUnit_p comment_unit;

        FunctionUnit()
            : LanguageUnit()
        {
        }

        string get_specialized_signature(string func_name = "")
        {
            NNFUSION_CHECK_NOT_NULLPTR(this->name_unit);
            NNFUSION_CHECK_NOT_NULLPTR(this->signature_unit);
            string fname = func_name == "" ? this->name_unit->get_code() : func_name;
            string sig = this->signature_unit->get_code();
            size_t pos = sig.find_last_of("(");
            NNFUSION_CHECK(pos > 0 && pos < sig.size());
            sig.insert(pos, fname);
            return sig;
        }

        string get_specialized_function_call(string func_name = "")
        {
            NNFUSION_CHECK_NOT_NULLPTR(this->name_unit);
            NNFUSION_CHECK_NOT_NULLPTR(this->call_unit);
            string fname = func_name == "" ? this->name_unit->get_code() : func_name;
            string call = this->call_unit->get_code();

            // size_t spos = call.find("cudaStream_t*");
            // if (spos == 0)
            // {
            //     size_t pos = call.find("(");
            //     call.insert(pos, fname);
            //     return call;
            // }
            return fname + call;
        }
    };

    using FunctionUnit_p = shared_ptr<FunctionUnit>;
}
