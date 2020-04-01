// Microsoft (c) 2019, Wenxiang
#pragma once
#include "nnfusion/common/languageunit.hpp"

#define LU_DECLARE(NAME) extern LanguageUnit_p NAME;

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#define LU_DEFINE(NAME, code)                                                                      \
    LanguageUnit_p NAME = LanguageUnit_p(new LanguageUnit(STR(NAME), code));

namespace nnfusion
{
    namespace kernels
    {
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