# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

ExternalProject_Add(
    ext_clang
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/clang/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/clang/stamp"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/clang"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/clang/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/clang"
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_clang SOURCE_DIR)
set(CLANG_SOURCE_DIR ${SOURCE_DIR})

ExternalProject_Add(
    ext_openmp
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/openmp/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/openmp/stamp"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/openmp"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/openmp/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/openmp"
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_openmp SOURCE_DIR)
set(OPENMP_SOURCE_DIR ${SOURCE_DIR})

if(DEFINED CMAKE_ASM_COMPILER)
    set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_ASM_COMPILER})
else()
    set(LLVM_CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
endif()

ExternalProject_Add(
    ext_llvm
    DEPENDS ext_clang ext_openmp
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_ASM_COMPILER=${LLVM_CMAKE_ASM_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX=${NNFUSION_THIRDPARTY_FOLDER}/build/llvm
                -DCMAKE_BUILD_TYPE=Release
                -DLLVM_ENABLE_ASSERTIONS=OFF
                -DLLVM_INCLUDE_TESTS=OFF
                -DLLVM_INCLUDE_EXAMPLES=OFF
                -DLLVM_BUILD_TOOLS=ON
                -DLLVM_TARGETS_TO_BUILD=X86
                -DLLVM_EXTERNAL_CLANG_SOURCE_DIR=${CLANG_SOURCE_DIR}
                -DLLVM_EXTERNAL_OPENMP_SOURCE_DIR=${OPENMP_SOURCE_DIR}
    UPDATE_COMMAND ""
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/stamp"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/llvm"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/llvm"
    BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMCore.a"
    EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Get_Property(ext_llvm INSTALL_DIR)

set(LLVM_LINK_LIBS
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangTooling.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangFrontendTool.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangFrontend.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangDriver.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangSerialization.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangCodeGen.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangParse.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangSema.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangStaticAnalyzerFrontend.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangStaticAnalyzerCheckers.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangStaticAnalyzerCore.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangAnalysis.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangARCMigrate.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangRewriteFrontend.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangEdit.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangAST.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangLex.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libclangBasic.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMLTO.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMPasses.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMObjCARCOpts.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMSymbolize.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDebugInfoPDB.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDebugInfoDWARF.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMMIRParser.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMCoverage.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMTableGen.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDlltoolDriver.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMOrcJIT.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMObjectYAML.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMLibDriver.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMOption.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86Disassembler.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86AsmParser.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86CodeGen.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMGlobalISel.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMSelectionDAG.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMAsmPrinter.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDebugInfoCodeView.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDebugInfoMSF.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86Desc.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMMCDisassembler.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86Info.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86AsmPrinter.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMX86Utils.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMMCJIT.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMLineEditor.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMInterpreter.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMExecutionEngine.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMRuntimeDyld.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMCodeGen.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMTarget.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMCoroutines.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMipo.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMInstrumentation.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMVectorize.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMScalarOpts.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMLinker.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMIRReader.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMAsmParser.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMInstCombine.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMTransformUtils.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMBitWriter.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMAnalysis.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMProfileData.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMObject.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMMCParser.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMMC.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMBitReader.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMCore.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMBinaryFormat.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMSupport.a
    ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/lib/libLLVMDemangle.a
)

if(APPLE)
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} curses z m)
else()
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} tinfo z m)
endif()

add_library(libllvm INTERFACE)
add_dependencies(libllvm ext_llvm)
target_include_directories(libllvm SYSTEM INTERFACE ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm/include)
target_link_libraries(libllvm INTERFACE ${LLVM_LINK_LIBS})
