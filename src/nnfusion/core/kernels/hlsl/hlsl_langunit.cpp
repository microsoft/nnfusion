// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hlsl_langunit.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::systems,
          R"(
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;

using Microsoft.Win32;
using System.Diagnostics;

)");

LU_DEFINE(header::D3D12APIWrapper, "#include \"D3D12APIWrapper.h\"\n");

// Declaration
LU_DEFINE(declaration::antares_hlsl_dll,
          R"(
[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxStreamCreate();

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamSubmit(IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamDestroy(IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamSynchronize(IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderLoad(string source, [Optional] int num_outputs, [Optional] int num_inputs);
[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderLoad(string source, out int num_outputs, out int num_inputs);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderUnload(IntPtr hShader);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxShaderGetProperty(IntPtr hShader, int arg_index, out long num_elements, out long type_size, out IntPtr dtype_name);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxShaderLaunchAsync(IntPtr hShader, IntPtr[] source, IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxEventCreate();

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxEventDestroy(IntPtr hEvent);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxEventRecord(IntPtr hEvent, IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern float dxEventElapsedTime(IntPtr hStart, IntPtr hStop);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxMemAlloc(long bytes);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemFree(IntPtr dptr);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemcpyHtoDAsync(IntPtr dptr, IntPtr hptr, long bytes, IntPtr hStream);

[DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemcpyDtoHAsync(IntPtr hptr, IntPtr dptr, long bytes, IntPtr hStream);

)");

LU_DEFINE(declaration::antares_hlsl_dll_cpp, R"(
HMODULE libtestdll;

int (*dxInit)(int flags);
void* (*dxStreamCreate)();
int (*dxStreamDestroy)(void* hStream);
int (*dxStreamSubmit)(void* hStream);
int (*dxStreamSynchronize)(void* hStream);
void* (*dxMemAlloc)(size_t bytes);
int (*dxMemFree)(void* dptr);
int (*dxMemcpyHtoDAsync)(void* dst, void* src, size_t bytes, void* hStream);
int (*dxMemcpyDtoHAsync)(void* dst, void* src, size_t bytes, void* hStream);
void* (*dxShaderLoad)(const char* src, int* num_inputs, int* num_outputs);
int (*dxShaderUnload)(void* hShader);
int (*dxShaderGetProperty)(void* hShader, int arg_index, size_t* num_elements, size_t* type_size, const char** dtype_name);
int (*dxShaderLaunchAsync)(void* hShader, void** buffers, void* hStream);
void* (*dxEventCreate)();
int (*dxEventRecord)(void* hEvent, void* hStream);
float (*dxEventElapsedTime)(void* hStart, void* hStop);
int (*dxEventDestroy)(void* hEvent);

)");
