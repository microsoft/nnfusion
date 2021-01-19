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

// Declaration
LU_DEFINE(declaration::antares_hlsl_dll_cs,
          R"(
public const string HlslDllName = @"antares_hlsl_v0.1_x64.dll";

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxStreamCreate();

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamSubmit(IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamDestroy(IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxStreamSynchronize(IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderLoad(string source, [Optional] int num_outputs, [Optional] int num_inputs);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderLoad(string source, out int num_outputs, out int num_inputs);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxShaderUnload(IntPtr hShader);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxShaderGetProperty(IntPtr hShader, int arg_index, out long num_elements, out long type_size, out IntPtr dtype_name);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxShaderLaunchAsync(IntPtr hShader, IntPtr[] source, IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxEventCreate();

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxEventDestroy(IntPtr hEvent);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxEventRecord(IntPtr hEvent, IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern float dxEventElapsedTime(IntPtr hStart, IntPtr hStop);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr dxMemAlloc(long bytes);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemFree(IntPtr dptr);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemcpyHtoDAsync(IntPtr dptr, IntPtr hptr, long bytes, IntPtr hStream);

[DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int dxMemcpyDtoHAsync(IntPtr hptr, IntPtr dptr, long bytes, IntPtr hStream);

[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool FreeLibrary(IntPtr hModule);

public static void UnloadHlslImportedDll()
{
    foreach (System.Diagnostics.ProcessModule mod in System.Diagnostics.Process.GetCurrentProcess().Modules)
    {
        if (mod.ModuleName == HlslDllName)
        {
            FreeLibrary(mod.BaseAddress);
        }
    }
}

)");
