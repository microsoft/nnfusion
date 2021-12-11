// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "nnfusion/common/languageunit.hpp"

#define LU_DECLARE(NAME) extern LanguageUnit_p NAME;

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE(NAME, code)                                                                      \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code));
#define LU_DEFINE_EXTEND(NAME, code, header, source)                                               \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code, header, source));
namespace nnfusion
{
    namespace kernels
    {
        namespace boilerplate
        {
            LU_DECLARE(MIT1);
            LU_DECLARE(MIT2);
        }
        namespace header
        {
            LU_DECLARE(stdio);
            // LU_DECLARE(algorithm);
            LU_DECLARE(fstream);
            LU_DECLARE(stdexcept);
            LU_DECLARE(sstream);
            LU_DECLARE(cmath);
            LU_DECLARE(assert);
            LU_DECLARE(vector);
            LU_DECLARE(cstring);
            LU_DECLARE(stdlib);
            LU_DECLARE(chrono);
            LU_DECLARE(ctime);
            LU_DECLARE(limits);
            LU_DECLARE(iostream);
            LU_DECLARE(windows);
            LU_DECLARE(unordered_map);
            LU_DECLARE(torch_extension);
        }

        namespace macro
        {
            LU_DECLARE(NNFUSION_DEBUG);
            LU_DECLARE(MIN);
        }

        namespace declaration
        {
            LU_DECLARE(typedef_int);
        }
    } // namespace kernels
} // namespace nnfusion