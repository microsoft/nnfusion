// Microsoft (c) 2019, Wenxiang Hu

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cuda_codegenerator.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/memory_allocator.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_bool(fcodegen_debug, false, "Add debug functions in Codegen-ed project.");
DEFINE_bool(fcodegen_timing, false, "Add timing functions in Codegen-ed project.");
DECLARE_bool(frt_const_folding);

namespace
{
    bool create_dir(std::string tar_path)
    {
        bool flag;
        int mkdir_status;
        struct stat s;
        int err = stat(tar_path.c_str(), &s);
        if (-1 == err)
        {
            mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (-1 == mkdir_status)
            {
                printf("Error creating directory: %s", (tar_path).c_str());
                flag = false;
            }
            else
                flag = true;
        }
        else
        {
            flag = true;
        }
        return flag;
    }

    bool save_file(LanguageUnit_p lu)
    {
        std::ofstream out(lu->symbol);
        out << lu->get_code();
        out.close();
        return true;
    }

} // namespace

// todo: add flags for future.
std::string CudaCodeGenerator::get_generate_cmakelists()
{
    LanguageUnit lu;
    lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")

link_directories(/usr/local/cuda/lib64)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
)" << (super_scaler_enable ? "find_package(MPI)" : "")
       << R"(
include_directories(${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} )"
       << (super_scaler_enable ? "${MPI_INCLUDE_PATH}" : "") << R"()

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
find_library(CUDA_cudart_LIBRARY libcudart.so /usr/local/cuda/lib64)
)" << (super_scaler_enable
           ? "find_library(SUPER_SCALER_LIBRARIES libsuper_scaler.so ${CMAKE_CURRENT_SOURCE_DIR})"
           : "")
       << R"(
cuda_add_library(nnfusion_naive_rt nnfusion_rt.cu)

target_link_libraries(nnfusion_naive_rt
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARIES})"
       << (super_scaler_enable ? R"(
    ${MPI_LIBRARIES}
    ${SUPER_SCALER_LIBRARIES}
    nccl)"
                               : "")
       << R"(
)

cuda_add_executable(main_test main_test.cpp)
target_link_libraries(main_test nnfusion_naive_rt cudnn culibos cublas)
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
else()
add_custom_command(
    TARGET nnfusion_naive_rt
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()
)";
    return lu.get_code();
}

void CudaCodeGenerator::post_projgen()
{
    // create_image_tests
    nnfusion::codegen::copy_file_from_templates("image_tests/image_test.cpp",
                                                "./image_tests/image_test.cpp");
    nnfusion::codegen::copy_file_from_templates("image_tests/CMakeLists_cuda.txt",
                                                "./image_tests/CMakeLists.txt");
}

std::string CudaCodeGenerator::get_target_name()
{
    return "cuda_codegen";
}

bool CudaCodeGenerator::setpwd()
{
    std::string working_dir = "./nnfusion_rt";
    std::string tar_path = working_dir + "/" + get_target_name() + "/";
    create_dir(working_dir);
    create_dir(tar_path);
    int status = chdir(tar_path.c_str());
    return (bool)status;
}

bool CudaCodeGenerator::projgen()
{
    save_file(this->lu_cmakefile);
    save_file(this->lu_nnfusion_rt);
    save_file(this->lu_header);
    save_file(this->lu_main);
    return true;
}

void CudaCodeGenerator::after_projgen()
{
    if (super_scaler_enable)
    {
        nnfusion::codegen::copy_file_from_templates("super_scaler/super_scaler.h",
                                                    "./super_scaler.h");
        LOG(WARNING) << "libsuper_scaler.so should be copied from "
                        "(build)/src/tools/nnfusion/templates/super_scaler/";
        nnfusion::codegen::copy_file_from_templates("super_scaler/libsuper_scaler.so",
                                                    "./libsuper_scaler.so");
    }
}

// bool CudaCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
//                         std::shared_ptr<TranslationUnit> tu)
// {
//     auto& p = tu->program;
//     for (auto iterator = p.entry; iterator != nullptr; iterator = iterator->next)
//     {
//         for (auto ins : *iterator)
//         {
//             LOG(INFO) << "instruction name: " << ins->name() << ", device: " << ins->Tag().Get<DeviceType>("Device");
//             //ins->Tag().Set<DeviceType>("Device", CUDA_GPU);
//         }
//     }
//     return true;
// }

std::vector<shared_ptr<const KernelRegistration>>
    CudaCodeGenerator::find_backend_kernels(const std::string& op_name,
                                            const shared_ptr<KernelContext>& ctx)
{
    return KernelRegistry::Global()->FindKernelRegistrations(op_name, CUDA_GPU, DT_FLOAT);
}

KernelEmitter::Pointer
    CudaCodeGenerator::match_kernel(std::vector<pair<DeviceType, KernelEmitter::Pointer>>& res)
{
    for (auto& k : res)
    {
        if (k.second != nullptr && k.first == device_type() &&
            k.second->get_or_emit_source() != nullptr)
        {
            return k.second;
        }
    }
    return nullptr;
}

bool CudaCodeGenerator::run(std::shared_ptr<InterpreterContext> ctx,
                            std::shared_ptr<TranslationUnit> tu)
{
    setpwd();

    auto& allocator_list = MemoryAllocatorFactory::get_allocator_list();
    auto async_manager = AsyncManagerFactory::get_async_manager(CUDA_GPU);

    this->lu_cmakefile = LanguageUnit_p(new LanguageUnit("CMakeLists.txt"));
    this->lu_nnfusion_rt = LanguageUnit_p(new LanguageUnit("nnfusion_rt.cu"));
    this->lu_header = LanguageUnit_p(new LanguageUnit("nnfusion_rt.h"));
    this->lu_main = LanguageUnit_p(new LanguageUnit("main_test.cpp"));

    LanguageUnit& lu_include = *this->lu_header;

    LanguageUnit lu_kernel_entry("KERNEL_ENTRY"); //func call for main()
    LanguageUnit lu_kernel_entry_header("KERNEL_ENTRY_HEADER");
    LanguageUnit lu_main_init("main_init");         //cuda_init()
    LanguageUnit lu_mem_plan_init("mem_plan_init"); // memory_pool planning in cuda_init()
    LanguageUnit lu_main_free("naive_free");        //cuda_free()

    bool rc = true;

    std::vector<shared_ptr<KernelEmitter>> kernels;
    std::unordered_map<KernelEmitter*, string> kernel_memcpy;
    auto& prog = tu->program;
    for (auto iterator : prog)
    {
        std::vector<shared_ptr<KernelEmitter>> block_kernels;
        for (auto ins : *iterator)
        {
            if (!(*ins)["Async_info"].is_valid())
            {
                CHECK_FAIL() << "Async info should be be assigned before this pass:"
                             << ins->getGNode()->get_name();
            }
            string op_name = ins->getGNode()->get_op_type();
            if (op_name == "Parameter")
            {
                continue;
            }
            if (ins->getGNode()->is_constant())
            {
                auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                    ins->getGNode()->get_op_type(), CUDA_GPU, DT_FLOAT);
                shared_ptr<KernelContext> ctx(new KernelContext(ins->getGNode()));
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();
                kernels.push_back(kernel);
                continue;
            }

            if ((*ins)["Kernel_Selection_Result"].is_valid())
            {
                auto res = (*ins)["Kernel_Selection_Result"]
                               .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                auto matched_kernel = match_kernel(res);
                if (matched_kernel)
                {
                    block_kernels.push_back(matched_kernel);

                    LanguageUnit m;
                    if ((*ins)["memcpy_pair"].is_valid())
                    {
                        auto mempairs = (*ins)["memcpy_pair"]
                                            .as<unordered_map<shared_ptr<descriptor::Tensor>,
                                                              shared_ptr<descriptor::Tensor>>>();
                        for (auto& it : mempairs)
                        {
                            string memcpykind = it.first->get_device_type() == GENERIC_CPU
                                                    ? "cudaMemcpyDeviceToHost"
                                                    : "cudaMemcpyHostToDevice";
                            m << "cudaMemcpy(" << it.first->get_name() << ", "
                              << it.second->get_name() << ", " << memcpykind << ");\n";
                        }
                        kernel_memcpy[matched_kernel.get()] = m.get_code();
                    }
                }
                else
                {
                    auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                        "AnyOP", CUDA_GPU, DT_FLOAT);
                    CHECK(kernel_reg != nullptr) << "AnyOp Kernel not found, op=" << op_name;
                    shared_ptr<KernelContext> ctx(new KernelContext(ins->getGNode()));
                    auto kernel = kernel_reg->m_factory(ctx);
                    kernel->get_or_emit_source();
                    block_kernels.push_back(kernel);
                }
            }
        }
        kernels.insert(kernels.end(), block_kernels.begin(), block_kernels.end());
    }

    LOG(INFO) << "Start dump whole source file...\n";
    // Code Gen
    LanguageUnit& lu = *this->lu_nnfusion_rt;
    lu << "// Microsoft (c) 2019, MSRA/NNFUSION Team\n";

    // Collect Requirement
    unordered_set<string> global_required;
    LanguageUnit re("REQUIREMENT");
    {
        re.require(header::assert);
        re.require(header::stdexcept);
        re.require(header::sstream);
        re.require(macro::CUDA_SAFE_CALL);
        //re.require(declaration::typedef_int);

        for (auto kernel : kernels)
        {
            if (!(kernel->is_emitted()))
            {
                return false;
            }

            for (auto& it : kernel->get_or_emit_source()->dep_unit->local_symbol)
            {
                re.require(it.second);
                global_required.insert(it.second->symbol);
            }
        }
    }

    lu << "#include \"nnfusion_rt.h\"\n\n";
    unordered_map<string, LanguageUnit_p> decleard_function_LU;
    std::unordered_map<shared_ptr<KernelEmitter>, string> kernel_stream;

    // Collect Function Definition
    {
        LanguageUnit def("FUNCTIONS");
        for (auto kernel : kernels)
        {
            //get kernel's stream name
            auto gnode = kernel->m_context->gnode;
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            if (async_info.execution_stream->is_default_stream())
                kernel_stream[kernel] = "0";
            else
                kernel_stream[kernel] = async_info.execution_stream->get_name();

            FunctionUnit_p fu = kernel->get_or_emit_source();
            for (auto& it : fu->body_unit->local_symbol)
            {
                if (it.second != fu->dep_unit)
                {
                    re.require(it.second);
                    global_required.insert(it.second->symbol);
                }
            }

            string func_key = fu->signature_unit->get_code() + fu->body_unit->get_code();
            if (kernel->is_static_function() ||
                decleard_function_LU.find(func_key) == decleard_function_LU.end())
            {
                std::string sig = fu->get_specialized_signature();
                int pos = sig.find(" __global__ "), next;
                if (pos >= 0)
                {
                    def << "\n";
                    def << fu->comment_unit->get_code();
                    def << sig << "\n";
                    def.block_begin();
                    def << fu->body_unit->get_code() << "\n";
                    def.block_end();
                    if (!kernel->is_static_function())
                    {
                        decleard_function_LU[func_key] = fu->name_unit;
                    }
                    while (pos < sig.size() && sig[pos] == ' ')
                        ++pos;
                    sig = sig.substr(pos + 12);
                    pos = sig.find("(");
                    if (pos >= 0)
                    {
                        std::string args = sig.substr(pos);
                        assert(args.size() > 0 && args[args.size() - 1] == ')');
                        args[args.size() - 1] = ',';

                        sig.insert(pos, "_Call");
                        sig.insert(pos + 6,
                                   "const dim3 &grids, const dim3 &blocks, unsigned mem, "
                                   "cudaStream_t stream, ");
                        def << "\n" << sig << "\n{\n";
                        def << "    return " << fu->name_unit->get_code()
                            << "<<<grids, blocks, mem, stream>>>(";

                        std::vector<std::string> params;
                        for (pos = 0; next = args.find(',', pos), next >= 0; pos = next + 1)
                        {
                            int start = next - 1;
                            while (start >= 0 && (isalpha(args[start]) || isdigit(args[start]) ||
                                                  args[start] == '_'))
                                --start;
                            params.push_back(args.substr(start + 1, next - start - 1));
                        }
                        def << join(params, ", ") << ");\n";
                        def << "}\n";
                    }
                }
                // add stream parameter
                else
                {
                    int pos = sig.find("(");
                    if (pos >= 0)
                    {
                        sig.insert(pos + 1, "cudaStream_t stream, ");
                    }
                    def << "\n";
                    def << fu->comment_unit->get_code();
                    def << sig << "\n";
                    def.block_begin();
                    def << fu->body_unit->get_code() << "\n";
                    def.block_end();
                    if (!kernel->is_static_function())
                    {
                        decleard_function_LU[func_key] = fu->name_unit;
                    }
                }
            }
            else
            {
                //def << "// Function declared:" << kernel->body_unit->symbol << "\n\n";
            }
        }

        //Write Dependency
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("header::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("macro::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        for (auto& it : re.local_symbol)
            if (it.second->symbol.find("declaration::") != string::npos)
                lu << it.second->get_code();
        lu << "\n";
        // stream and event declaration
        if (async_manager->num_stream() > 0)
            lu << async_manager->emit_stream_decl()->get_code();
        if (async_manager->num_event() > 0)
            lu << async_manager->emit_event_decl()->get_code();
        //Write Code
        lu << def.collect_code() << "\n";
    }

    bool enable_debug = FLAGS_fcodegen_debug;
    bool enable_timing = FLAGS_fcodegen_timing;
    bool enable_rt_const_folding = FLAGS_frt_const_folding;

    // Generate graph configs
    {
        lu_kernel_entry << "\n#ifndef __NNFUSION_GRAPH_CONFIG__\n";
        lu_kernel_entry << "#define __NNFUSION_GRAPH_CONFIG__\n";
        lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_NUM " << tu->arg.size() << "\n";
        lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_NUM " << tu->out.size() << "\n";
        for (int i = 0; i < tu->arg.size(); i++)
        {
            lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_DTYPE_" << i << " "
                            << tu->arg[i]->get_element_type().c_type_string() << "\n";
            lu_kernel_entry << "#define NNFUSION_GRAPH_INPUT_SHAPE_" << i << " {";
            auto& shape = tu->arg[i]->get_shape();
            for (int j = 0; j < shape.size(); ++j)
            {
                if (j > 0)
                    lu_kernel_entry << ", ";
                lu_kernel_entry << shape[j];
            }
            lu_kernel_entry << "}\n";
        }
        for (int i = 0; i < tu->out.size(); i++)
        {
            lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_DTYPE_" << i << " "
                            << tu->out[i]->get_element_type().c_type_string() << "\n";
            lu_kernel_entry << "#define NNFUSION_GRAPH_OUTPUT_SHAPE_" << i << " {";
            auto& shape = tu->out[i]->get_shape();
            for (int j = 0; j < shape.size(); ++j)
            {
                if (j > 0)
                    lu_kernel_entry << ", ";
                lu_kernel_entry << shape[j];
            }
            lu_kernel_entry << "}\n";
        }
        lu_kernel_entry << "#endif\n\n";
    }

    // Generate caller function body
    {
        unordered_set<string> allocated;

        lu_kernel_entry << "extern \"C\" int kernel_entry(";
        // Add param
        {
            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "** " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            lu_kernel_entry << join(params, ", ");
        }
        lu_kernel_entry << ")";
        lu_kernel_entry_header << lu_kernel_entry.get_code();
        lu_kernel_entry << "\n";
        lu_kernel_entry.block_begin();

        //Planning
        for (const auto& allocator : allocator_list)
        {
            lu_mem_plan_init << allocator.second->emit_memory_init()->get_code();
            lu_main_init << allocator.second->emit_memory_alloc()->get_code();
        }
        //Function Call
        {
            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu_main_init << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu_main_init << "CUDNN_SAFE_CALL(cudnnCreate(&global_cudnn_handle));\n";
            }
            if (global_required.count("declaration::num_SMs") > 0)
            {
                lu_main_init << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                                "cudaDevAttrMultiProcessorCount, 0));\n";
            }
            if (async_manager->num_stream() > 0)
                lu_main_init << async_manager->emit_stream_init()->get_code();
            if (async_manager->num_event() > 0)
                lu_main_init << async_manager->emit_event_init()->get_code();
            if (enable_timing)
            {
                lu_kernel_entry << "static cudaEvent_t hEvents[" << kernels.size() << "];\n";
                lu_kernel_entry << "size_t eventCnt = 0;\n";
                lu_kernel_entry << "if (!hEvents[eventCnt]) "
                                   "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                lu_kernel_entry << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
            }
            //const
            set<string> constant_vals;
            size_t kernel_order = 0;
            for (auto kernel : kernels)
            {
                auto gnode = kernel->m_context->gnode;
                auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
                FunctionUnit_p fu = kernel->get_or_emit_source();
                std::string func_name;
                if (kernel->is_static_function())
                {
                    func_name = fu->name_unit->get_code();
                }
                else
                {
                    string func_key = fu->signature_unit->get_code() + fu->body_unit->get_code();
                    CHECK(decleard_function_LU.find(func_key) != decleard_function_LU.end());
                    func_name = decleard_function_LU[func_key]->get_code();
                }

                if (func_name.compare(0, 9, "Constant_") == 0)
                {
                    CHECK(async_info.execution_stream->is_default_stream())
                        << "Kernel function calls in cuda_init() should use default stream.";
                    std::string function_call = fu->call_unit->get_code();
                    // add stream info
                    int pos = function_call.find("(");
                    if (pos >= 0)
                    {
                        std::string stream_name = kernel_stream[kernel] + ", ";
                        function_call.insert(pos + 1, stream_name);
                    }
                    // if (!async_info.wait_events.empty())
                    // {
                    //     for (auto event : async_info.wait_events)
                    //     {
                    //         lu_main_init
                    //             << async_manager->emit_event_wait(async_info.execution_stream, event)
                    //                     ->get_code();
                    //     }
                    // }
                    lu_main_init << func_name << function_call;
                    // if (async_info.record_event != nullptr)
                    // {
                    //     lu_main_init
                    //         << async_manager->emit_event_record(async_info.record_event)->get_code();
                    // }

                    if (enable_rt_const_folding)
                    {
                        for (auto& out : kernel->m_context->output_names)
                            constant_vals.insert(out);
                    }
                }
                else
                {
                    // Put constant-able node into cuda_init();
                    bool const_inputs = true;
                    for (auto& in : kernel->m_context->input_names)
                    {
                        if (constant_vals.find(in) == constant_vals.end())
                        {
                            const_inputs = false;
                            break;
                        }
                    }
                    if (const_inputs)
                    {
                        for (auto& out : kernel->m_context->output_names)
                        {
                            constant_vals.insert(out);
                        }
                    }

                    auto& call_place = const_inputs ? lu_main_init : lu_kernel_entry;
                    if (const_inputs)
                        CHECK(async_info.execution_stream->is_default_stream())
                            << "Kernel function calls in cuda_init() should use default stream.";

                    const string node_name = (kernel->m_context->gnode)
                                                 ? kernel->m_context->gnode->get_name()
                                                 : "internal_node";
                    if (!const_inputs)
                        call_place << " // order=" << ++kernel_order << ", name=" << node_name
                                   << "\n";
                    // Put memcpy here
                    lu_kernel_entry << kernel_memcpy[kernel.get()];

                    std::string function_call = fu->get_specialized_funciton_call(func_name);
                    int pos_left = function_call.find("<<<"),
                        pos_right = function_call.find(">>>(");
                    if (pos_left >= 0 && pos_right >= 0)
                    {
                        std::string builder = function_call.substr(0, pos_left) + "_Call(";
                        builder +=
                            function_call.substr(pos_left + 3, pos_right - pos_left - 3) + ", ";
                        // add stream info
                        builder += kernel_stream[kernel] + ", ";
                        builder += function_call.substr(pos_right + 4);
                        function_call = std::move(builder);
                    }
                    // add stream for cudalib emitter function call
                    else
                    {
                        int pos = function_call.find("(");
                        std::string stream_name = kernel_stream[kernel] + ", ";
                        function_call.insert(pos + 1, stream_name);
                    }

                    if (!async_info.wait_events.empty())
                    {
                        for (auto event : async_info.wait_events)
                        {
                            call_place << async_manager
                                              ->emit_event_wait(async_info.execution_stream, event)
                                              ->get_code();
                        }
                    }
                    call_place << function_call;
                    if (async_info.record_event != nullptr)
                    {
                        call_place << async_manager->emit_event_record(async_info.record_event)
                                          ->get_code();
                    }

                    if (enable_debug && !const_inputs)
                    {
                        for (size_t i = 0; i < kernel->m_context->outputs.size(); i++)
                        {
                            if (kernel->m_context->outputs[i]->get_element_type().c_type_string() !=
                                "float")
                                continue;
                            auto out_name = kernel->m_context->output_names[i];
                            call_place << "Debug(\"" << node_name << ", " << out_name << "\", "
                                       << out_name << ", \"" << join(kernel->m_context->input_names)
                                       << "\", " << kernel->m_context->outputs[i]->size(false)
                                       << ");\n";
                        }
                    }
                    if (enable_timing && !const_inputs)
                    {
                        lu_kernel_entry << "if (!hEvents[eventCnt]) "
                                           "CUDA_SAFE_CALL(cudaEventCreate(&hEvents[eventCnt]));\n";
                        lu_kernel_entry
                            << "CUDA_SAFE_CALL(cudaEventRecord(hEvents[eventCnt++]));\n";
                    }
                }
            }
            // destroy cuda stream and event
            if (async_manager->num_stream() > 0)
                lu_main_free << async_manager->emit_stream_destroy()->get_code();
            if (async_manager->num_event() > 0)
                lu_main_free << async_manager->emit_event_destroy()->get_code();

            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu_main_free << "CUBLAS_SAFE_CALL(cublasDestroy(global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu_main_free << "CUDNN_SAFE_CALL(cudnnDestroy(global_cudnn_handle));\n";
            }
            if (global_required.count("header::super_scaler"))
            {
                lu_kernel_entry << "super_scaler_sync();\n";
                lu_main_free << "super_scaler_finalization();\n";
            }
            if (enable_timing)
            {
                lu_kernel_entry << "// Output Timing Result:\n";
                lu_kernel_entry << "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n";
                lu_kernel_entry << "float total_ms = 0;\n";
                lu_kernel_entry
                    << "for (size_t i = 1; i < eventCnt; ++i) { float ms; "
                       "CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, hEvents[i - 1], hEvents[i])); "
                       "printf(\"%zd: %.2f ms\\n\", i, ms), total_ms += ms; }\n";
                lu_kernel_entry
                    << "printf(\"Timing total (except outer memcpy) = %g ms.\\n\", total_ms);\n";
            }
        }
        for (const auto& allocator : allocator_list)
        {
            lu_main_free << allocator.second->emit_memory_free()->get_code();
        }

        lu_kernel_entry << "return 0;\n";
        lu_kernel_entry.block_end();
    }

    lu << "\n";
    {
        lu << lu_mem_plan_init.get_code();
        lu << "\nextern \"C\" void cuda_init()";
        lu.block_begin();
        {
            lu << "CUDA_SAFE_CALL(cudaDeviceReset());\n";
            if (global_required.count("header::super_scaler"))
                lu << "super_scaler_initialization();\n";
            else
                lu << "CUDA_SAFE_CALL(cudaSetDevice(0));\n";
            lu << lu_main_init.get_code();
        }
        lu.block_end();

        lu << "extern \"C\" void cuda_free()";
        lu.block_begin();
        {
            lu << lu_main_free.get_code();
        }

        lu.block_end();
    }
    lu << "\n";

    if (enable_debug)
    {
        lu << R"(
     inline void Debug(std::string name, float* tensor_ptr, std::string inputs, size_t debug_size = 10, size_t offset=0)
     {
         float* host_tensor = (float*)malloc(sizeof(float) * debug_size);
         CUDA_SAFE_CALL(cudaDeviceSynchronize());
         CUDA_SAFE_CALL(cudaMemcpy(host_tensor, tensor_ptr + offset,  sizeof(float) * debug_size, cudaMemcpyDeviceToHost));
         CUDA_SAFE_CALL(cudaDeviceSynchronize());
         size_t print_size = min((size_t)10, debug_size);
         printf("%s: ", name.c_str());
         for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i]);
         printf("...(size= %lu end with %e ) :", debug_size, host_tensor[debug_size - 1]);
         //print with an offset
         size_t print_offset = debug_size / 3;
         print_size = min((size_t)10, debug_size - print_offset);
         for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i + print_offset]);
         printf("...(offset= %lu) ", print_offset);
         printf(": %s\n", inputs.c_str());
     }
             )";
    }
    lu << lu_kernel_entry.get_code() << "\n\n";

    // // Test function
    // {
    //     lu << "extern \"C\" int naive_test(";
    //     // Add param
    //     {
    //         vector<string> params;
    //         for (int i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto tv = tu->arg[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << type << "* " << tv->get_name() << "_host";
    //             params.push_back(ss.str());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++)
    //         {
    //             auto tv = tu->out[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << type << "* " << tv->get_name() << "_host";
    //             params.push_back(ss.str());
    //         }

    //         lu << join(params, ", ");
    //     }
    //     lu << ")\n";
    //     lu.block_begin();
    //     {
    //         for (size_t i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tensor = *tu->arg[i];
    //             lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
    //                << ";\n"
    //                << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
    //                << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << "));\n";

    //             lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name()
    //                << "_host, " << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << ", "
    //                << "cudaMemcpyHostToDevice));\n";
    //         }

    //         lu << "//output arguments\n";
    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
    //                << ";\n"
    //                << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
    //                << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << "));\n";
    //         }

    //         vector<string> params;
    //         for (int i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tv = tu->arg[i];
    //             params.push_back(tv->get_name());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tv = tu->out[i];
    //             params.push_back(tv->get_name());
    //         }

    //         lu << "naive_entry(" << join(params, ", ") << ");\n";

    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
    //                << tensor.get_name() << ", " << tensor.get_tensor_layout()->get_size() << " * "
    //                << tensor.get_element_type().size() << ", "
    //                << "cudaMemcpyDeviceToHost));\n";
    //         }

    //         for (size_t i = 0; i < tu->arg.size(); i++)
    //         {
    //             auto& tensor = *tu->arg[i];
    //             lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
    //         }

    //         for (size_t i = 0; i < tu->out.size(); i++)
    //         {
    //             auto& tensor = *tu->out[i];
    //             lu << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
    //         }
    //     }
    //     lu << "return 0;\n";
    //     lu.block_end();
    // }

    // lu << "\n";

    // // Test function 2
    // {
    //     lu << "extern \"C\" int naive_test_simple(void** args)\n";
    //     // Add param
    //     lu.block_begin();
    //     {
    //         lu << "return naive_test(";
    //         vector<string> params;
    //         int acc = 0;
    //         for (int i = 0; i < tu->arg.size(); i++, acc++)
    //         {
    //             auto tv = tu->arg[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << "(" << type << "*)args[" << acc << "]";
    //             params.push_back(ss.str());
    //         }

    //         for (int i = 0; i < tu->out.size(); i++, acc++)
    //         {
    //             auto tv = tu->out[i];
    //             string type = tv->get_element_type().c_type_string();
    //             stringstream ss;
    //             ss << "(" << type << "*)args[" << acc << "]";
    //             params.push_back(ss.str());
    //         }
    //         lu << join(params, ", ");
    //         lu << ");\n";
    //     }
    //     lu.block_end();
    // }

    // generate main() function
    std::string function_include =
        "#include \"nnfusion_rt.h\"\n#include <stdlib.h>\n#include <stdio.h>\n";
    LanguageUnit h2dcopy("h2dcopy");
    LanguageUnit d2hcopy("d2hcopy");
    LanguageUnit fillval("fillval");

    LanguageUnit& lu_main = *this->lu_main;
    {
        lu_main << function_include << "\n";
        lu_main << header::stdexcept->get_code();
        lu_main << "#include <sstream>\n";
        lu_main << "#include <cuda_profiler_api.h>\n";
        lu_main << macro::CUDA_SAFE_CALL->get_code();
        lu_main << "\n";
        lu_main << "int main(void)";
        lu_main.block_begin();
        {
            lu_main << "\ncuda_init();\n\n";

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                //malloc host input arg
                lu_main << "//input argument\n";
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host, *" << tensor.get_name() << ";\n";

                lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                        << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ")* "
                        << tensor.get_tensor_layout()->get_size() << "));\n";
                lu_main << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ", "
                        << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << "));\n";

                fillval << "for (int i = 0; i < " << tensor.get_tensor_layout()->get_size()
                        << "; ++i) " << tensor.get_name() << "_host[i] = 1.0f;\n";

                h2dcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", "
                        << tensor.get_name() << "_host, "
                        << "sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << ", "
                        << "cudaMemcpyHostToDevice));\n";
            }

            lu_main << "\n//output arguments\n";
            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                //malloc host output arg
                lu_main << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                        << "_host, *" << tensor.get_name() << ";\n";

                lu_main << "CUDA_SAFE_CALL(cudaMallocHost((void**)&" << tensor.get_name()
                        << "_host, sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << "));\n";

                d2hcopy << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                        << tensor.get_name() << ", "
                        << " sizeof(" << tensor.get_element_type().c_type_string() << ") * "
                        << tensor.get_tensor_layout()->get_size() << ", "
                        << "cudaMemcpyDeviceToHost));\n";
            }

            lu_main << "\n// fill input values\n";
            lu_main << fillval.get_code() << "\n";
            lu_main << h2dcopy.get_code() << "\n";

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name());
            }
            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back("&" + tv->get_name());
            }
            int warm_step = 5, test_step = 100;
            if (enable_debug)
            {
                warm_step = 0;
                test_step = 1;
            }
            lu_main << "\n//warm up for 5 iters:\n";
            lu_main << "for(int i_=0; i_< " << warm_step << "; i_++)\n";
            lu_main.block_begin();
            lu_main << h2dcopy.get_code();
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";
            lu_main << d2hcopy.get_code();
            lu_main << "CUDA_SAFE_CALL(cudaDeviceSynchronize()); \n";

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu_main << "printf(\"%s \\n\", \"" << tensor.get_name() << ":\");\n"
                        << "for (int i = 0; i < "
                        << std::min(size_t(10), tensor.get_tensor_layout()->get_size())
                        << "; ++i) printf(\"%e \", (float)" << tensor.get_name() << "_host[i]); "
                        << "\nprintf(\" .. (size = " << tensor.get_tensor_layout()->get_size()
                        << ", ends with %e);\\n\", (float)" << tensor.get_name() << "_host["
                        << tensor.get_tensor_layout()->get_size() - 1 << "]);\n";
            }
            lu_main.block_end();

            lu_main << "\n//GPU time measurement\n";
            lu_main << "cudaEvent_t start, stop;\n";
            lu_main << "cudaEventCreate(&start);\n";
            lu_main << "cudaEventCreate(&stop);\n";

            lu_main << "\n//time measurement\n";
            lu_main << "cudaEventRecord(start);\n\n";
            lu_main << "//kernel call\n";

            lu_main << "int steps = " << test_step << ";\n";
            lu_main << "cudaProfilerStart();\n";
            lu_main << "for (int i_=0; i_<steps; i_++)\n";
            lu_main.block_begin();

            lu_main << h2dcopy.get_code();
            // kernel launch
            lu_main << "kernel_entry(" << join(params, ", ") << ");\n";

            // lu_main << d2hcopy.get_code();

            lu_main.block_end();
            lu_main << "cudaProfilerStop();\n";

            lu_main << "//time measurement\n";
            lu_main << "\ncudaEventRecord(stop);\n";
            lu_main << "cudaEventSynchronize(stop);\n";
            lu_main << "float milliseconds = 0;\n";
            lu_main << "cudaEventElapsedTime(&milliseconds, start, stop);\n";
            lu_main << "printf(\"function execution time: %f ms\\n\", milliseconds/steps);\n";
            lu_main << "\n//free context\n";

            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu_main << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
            }

            lu_main << "cuda_free();\n\n";
        }

        //free host input args
        for (size_t i = 0; i < tu->arg.size(); i++)
        {
            auto& tensor = *tu->arg[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }
        //free host output args
        for (size_t i = 0; i < tu->out.size(); i++)
        {
            auto& tensor = *tu->out[i];
            lu_main << "cudaFreeHost(" << tensor.get_name() << "_host);\n";
        }

        lu_main << "return 0;\n";
        lu_main.block_end();
    }

    //generate include header file
    lu_include << "// Microsoft (c) 2019\n";
    lu_include << "#pragma once\n";
    lu_include << declaration::typedef_int->get_code() << "\n";
    lu_include << lu_kernel_entry_header.get_code() << ";\n";
    lu_include << "extern \"C\" void cuda_init();\n";
    lu_include << "extern \"C\" void cuda_free();\n";
    lu_include << header::cuda->get_code();

    //generate CMakeList.txt
    LanguageUnit& lu_cmake = *this->lu_cmakefile;
    super_scaler_enable = global_required.count("header::super_scaler") > 0;
    lu_cmake << get_generate_cmakelists();

    post_projgen();
    projgen();
    after_projgen();

    // change to working directory
    int status = chdir("../../");
    return rc;
}
