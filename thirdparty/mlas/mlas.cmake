# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(CheckCXXCompilerFlag)

set(mlas_common_srcs
  ${CMAKE_CURRENT_LIST_DIR}/lib/platform.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/threading.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/dgemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/sgemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/qgemm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/convolve.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/pooling.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/reorder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/snchwc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/activate.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/logistic.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/tanh.cpp
  ${CMAKE_CURRENT_LIST_DIR}/lib/erf.cpp
)

if(MSVC)
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(asm_filename ${CMAKE_CURRENT_LIST_DIR}/lib/arm64/SgemmKernelNeon.asm)
    set(pre_filename ${CMAKE_CURRENT_BINARY_DIR}/SgemmKernelNeon.i)
    set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/SgemmKernelNeon.obj)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(ARMASM_FLAGS "-g")
    else()
      set(ARMASM_FLAGS "")
    endif()

    add_custom_command(
      OUTPUT ${obj_filename}
        COMMAND
            cl.exe /P ${asm_filename}
        COMMAND
            armasm64.exe ${ARMASM_FLAGS} ${pre_filename} ${obj_filename}
    )
    set(mlas_platform_srcs ${obj_filename})
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM" OR CMAKE_GENERATOR MATCHES "ARM")
    set(mlas_platform_srcs
      ${CMAKE_CURRENT_LIST_DIR}/lib/arm/sgemmc.cpp
    )
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64" OR CMAKE_GENERATOR MATCHES "Win64")
    enable_language(ASM_MASM)

    set(mlas_platform_srcs
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8S8KernelAvx2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemvU8S8KernelAvx2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8S8KernelAvx512BW.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemvU8S8KernelAvx512BW.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8S8KernelAvx512Vnni.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemvU8S8KernelAvx512Vnni.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8U8KernelAvx2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8U8KernelAvx512BW.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/QgemmU8U8KernelAvx512Vnni.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/DgemmKernelSse2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/DgemmKernelAvx.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/DgemmKernelFma3.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/DgemmKernelAvx512F.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SgemmKernelSse2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SgemmKernelAvx.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SgemmKernelM1Avx.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SgemmKernelFma3.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SgemmKernelAvx512F.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SconvKernelSse2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SconvKernelAvx.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SconvKernelFma3.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SconvKernelAvx512F.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SpoolKernelSse2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SpoolKernelAvx.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/SpoolKernelAvx512F.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/sgemma.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/cvtfp16a.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/LogisticKernelFma3.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/TanhKernelFma3.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/amd64/ErfKernelFma3.asm
    )
  else()
    enable_language(ASM_MASM)

    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")

    set(mlas_platform_srcs
      ${CMAKE_CURRENT_LIST_DIR}/lib/i386/SgemmKernelSse2.asm
      ${CMAKE_CURRENT_LIST_DIR}/lib/i386/SgemmKernelAvx.asm
    )
  endif()
else()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(ARM TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
      set(ARM64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
      set(X86_64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
      set(X86 TRUE)
    endif()
  else()
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -dumpmachine
      OUTPUT_VARIABLE dumpmachine_output
      ERROR_QUIET
    )
    if(dumpmachine_output MATCHES "^arm.*")
      set(ARM TRUE)
    elseif(dumpmachine_output MATCHES "^aarch64.*")
      set(ARM64 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      set(X86_64 TRUE)
    endif()
  endif()

  if(ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

    set(mlas_platform_srcs
      ${CMAKE_CURRENT_LIST_DIR}/lib/arm/sgemmc.cpp
    )
  elseif(ARM64)
    enable_language(ASM)

    set(mlas_platform_srcs
      ${CMAKE_CURRENT_LIST_DIR}/lib/aarch64/SgemmKernelNeon.S
    )
  elseif(X86)
    enable_language(ASM)

    set(mlas_platform_srcs_sse2
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86/SgemmKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86/SgemmKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
    )
  elseif(X86_64)
    enable_language(ASM)

    # The LLVM assembler does not support the .arch directive to enable instruction
    # set extensions and also doesn't support AVX-512F instructions without
    # turning on support via command-line option. Group the sources by the
    # instruction set extension and explicitly set the compiler flag as appropriate.

    set(mlas_platform_srcs_sse2
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/DgemmKernelSse2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelSse2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmTransposePackB16x4Sse2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SconvKernelSse2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SpoolKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/DgemmKernelAvx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelAvx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelM1Avx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelM1TransposeBAvx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmTransposePackB16x4Avx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SconvKernelAvx.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SpoolKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs_avx2
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8S8KernelAvx2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemvU8S8KernelAvx2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8U8KernelAvx2.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/DgemmKernelFma3.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelFma3.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SconvKernelFma3.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/LogisticKernelFma3.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/TanhKernelFma3.S
      ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/ErfKernelFma3.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

    # Some platforms do not support AVX512 flags but still able to compile the source
    # Others support the flag and refuse to compile without the flag.
    # We have to run all 3 checks
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
    if(HAS_AVX512F)
      set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    else()
      set(CMAKE_REQUIRED_FLAGS "")
    endif()

    check_cxx_source_compiles("
      int main() {
        asm(\"vpxord %zmm0,%zmm0,%zmm0\");
        return 0;
      }"
      AVX512F_COMPILES
    )

    if(AVX512F_COMPILES)
      set(mlas_platform_srcs_avx512f
        ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/DgemmKernelAvx512F.S
        ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SgemmKernelAvx512F.S
        ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SconvKernelAvx512F.S
        ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/SpoolKernelAvx512F.S
      )
      if(HAS_AVX512F)
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")
      endif()
      
      # AVX512BW support is only available if AVX512F support is present.
      check_cxx_compiler_flag("-mavx512bw" HAS_AVX512BW)
      if(HAS_AVX512BW)
        set(CMAKE_REQUIRED_FLAGS "-mavx512bw")
      endif()
      check_cxx_source_compiles("
        int main() {
          asm(\"vpmaddwd %zmm0,%zmm0,%zmm0\");
          return 0;
        }"
        AVX512BW_COMPILES
      )
      
      if(AVX512BW_COMPILES)
        set(mlas_platform_srcs_avx512bw
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8S8KernelAvx512BW.S
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemvU8S8KernelAvx512BW.S
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8S8KernelAvx512Vnni.S
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemvU8S8KernelAvx512Vnni.S
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8U8KernelAvx512BW.S
          ${CMAKE_CURRENT_LIST_DIR}/lib/x86_64/QgemmU8U8KernelAvx512Vnni.S
        )
        
        if(HAS_AVX512BW)
          set_source_files_properties(${mlas_platform_srcs_avx512bw} PROPERTIES COMPILE_FLAGS "-mavx512bw")
        endif()
      else() # AVX512BW_COMPILES
        # 
        set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512BW_UNSUPPORTED")
      endif() # AVX512BW_COMPILES
    else() # AVX512F_COMPILES
      set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512F_UNSUPPORTED")
    endif() # AVX512F_COMPILES

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${mlas_platform_srcs_avx512f}
      ${mlas_platform_srcs_avx512bw}
    )
  endif()
endif()

add_library(mlas STATIC ${mlas_common_srcs} ${mlas_platform_srcs})
target_link_libraries(mlas threadpool)
target_include_directories(mlas PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc ${CMAKE_CURRENT_LIST_DIR}/lib ${CMAKE_CURRENT_LIST_DIR}/lib/amd64)
target_include_directories(mlas SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/inc)
set_target_properties(mlas PROPERTIES FOLDER "mlas")
