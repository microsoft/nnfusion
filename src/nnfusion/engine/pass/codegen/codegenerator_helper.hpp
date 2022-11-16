// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"

// #define __USING_HOST_CALL_FORMAT___

namespace nnfusion
{
    namespace codegenerator
    {
        nnfusion::LanguageUnit_p extern_function(nnfusion::LanguageUnit_p lu);
        nnfusion::LanguageUnit_p extern_variable(nnfusion::LanguageUnit_p lu);

        class FunctionFile : public LanguageUnit
        {
        public:
            using FunctionFile_p = shared_ptr<FunctionFile>;
            static FunctionFile_p
                convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel);
            FunctionFile(string extern_declare, LanguageUnit_p file_context);
            FunctionFile() { extern_declare = ""; }
            string get_extern_declare() { return extern_declare; };
            virtual void save_file();
            void merge_from(FunctionFile_p func);

        private:
            string extern_declare;
            string suffix_str = ".cu";

            /*
            Original FunctionUnit includes:
            LanguageUnit_p name_unit;
            LanguageUnit_p signature_unit; // void (float* input0, float* input1, float* output0)
            LanguageUnit_p body_unit;
            LanguageUnit_p dep_unit;
            LanguageUnit_p call_unit; // (tensor1, tensor2, tensor3)
            LanguageUnit_p comment_unit

            Based on the profiler's codegen:
                1. put dep_unit into extern; ------------------> in to function.cu
                2. put sig & body into single file; -----------^
                3. generate extern function def from sig; -----> Replacing original function def

            In the cmakelist files:
                1. Compiling <function>.cu in to objects <function>;
                2. Compiling nnfusion_rt.cu/hpp into objects "nnfusion_rt";
                3. Link them together.
            */

            //\todo: IS THIS WAY GENERAL? Any C-series(*cc) compiler will support this way.
        };
        class CPUFunctionFile : public FunctionFile
        {
        public:
            using CPUFunctionFile_p = shared_ptr<CPUFunctionFile>;
            static CPUFunctionFile_p
                convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel);
            CPUFunctionFile(string extern_declare, LanguageUnit_p file_context)
                : FunctionFile(extern_declare, file_context)
            {
            }
            CPUFunctionFile() { extern_declare = ""; }
            void save_file() override;

        private:
            string extern_declare;
            string suffix_str = ".cpp";
        };

        using FunctionFile_p = FunctionFile::FunctionFile_p;
        using CPUFunctionFile_p = CPUFunctionFile::CPUFunctionFile_p;

        class HLSLFunctionFile : public FunctionFile
        {
        public:
            using HLSLFunctionFile_p = shared_ptr<HLSLFunctionFile>;
            static HLSLFunctionFile_p
                convert_from(std::shared_ptr<nnfusion::kernels::KernelEmitter> kernel);
            HLSLFunctionFile(string extern_declare, LanguageUnit_p file_context)
                : FunctionFile(extern_declare, file_context)
            {
            }
            HLSLFunctionFile() { extern_declare = ""; }
            void save_file() override { return; } // not support yet
        private:
            string extern_declare;
            string suffix_str = ".hlsl";
        };

        using FunctionFile_p = FunctionFile::FunctionFile_p;
        using HLSLFunctionFile_p = HLSLFunctionFile::HLSLFunctionFile_p;
    } // namespace codegenerator
} // namespace nnfusion
