
#include "common_langunit.hpp"

using namespace nnfusion::kernels;

// Boilderplate
LU_DEFINE(boilerplate::MIT1,
          "// Copyright (c) Microsoft Corporation.\n// Licensed under the MIT License.\n\n");
LU_DEFINE(boilerplate::MIT2,
          "# Copyright (c) Microsoft Corporation.\n# Licensed under the MIT License.\n\n");
// Header
LU_DEFINE(header::stdio, "#include <stdio.h>\n");
LU_DEFINE(header::cmath, "#include <cmath>\n");
// LU_DEFINE(header::algorithm, "#include <algorithm>\n");
LU_DEFINE(header::fstream, "#include <fstream>\n");
LU_DEFINE(header::stdexcept, "#include <stdexcept>\n");
LU_DEFINE(header::sstream, "#include <sstream>\n");
LU_DEFINE(header::assert, "#include <assert.h>\n");
LU_DEFINE(header::vector, "#include <vector>\n");
LU_DEFINE(header::cstring, "#include <cstring>\n");
LU_DEFINE(header::stdlib, "#include <stdlib.h>\n");
LU_DEFINE(header::chrono, "#include <chrono>\n");
LU_DEFINE(header::ctime, "#include <ctime>\n");
LU_DEFINE(header::limits, "#include <limits>\n");
LU_DEFINE(header::iostream, "#include <iostream>\n");
LU_DEFINE(header::windows, "#define NOMINMAX\n#include <windows.h>\n");
LU_DEFINE(header::unordered_map, "#include <unordered_map>\n");
LU_DEFINE(header::torch_extension, "#include <torch/extension.h>\n");

// Macro
LU_DEFINE(macro::NNFUSION_DEBUG, "#define NNFUSION_DEBUG\n");
LU_DEFINE(macro::MIN, "#define MIN(a,b) ((a)>(b)?(b):(a))\n");

// Declaration
// LU_DEFINE(declaration::typedef_int, "#include <stdint.h>\n");
LU_DEFINE(declaration::typedef_int,
          R"(
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef signed long int int64_t;
#define __int8_t_defined
)");