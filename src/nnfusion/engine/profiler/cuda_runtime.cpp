// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#include "cuda_runtime.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion::profiler;
using namespace nnfusion::kernels;
using namespace nnfusion::kernels::cuda;

bool CudaDefaultRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;

    // assign async info
    auto async_manager =
        nnfusion::async::AsyncManagerFactory::get_device_stream_async_manager(nullptr, CUDA_GPU);
    auto gnode = ke->kernel->m_context->gnode;
    (*gnode)["Async_info"] = nnfusion::async::AsyncExecutionInfo();
    auto& async_info = (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
    async_info.execution_stream = async_manager->set_stream();
    bool require_cudnn_handle = false;
    bool require_cublas_handle = false;
    if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(ke->kernel))
    {
        if (kernel->require_cudnn_handle())
        {
            async_info.execution_stream->add_binding_symbol("cudnn_handle");
            require_cudnn_handle = true;
        }
        if (kernel->require_cublas_handle())
        {
            async_info.execution_stream->add_binding_symbol("cublas_handle");
            require_cublas_handle = true;
        }
    }

    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code() + ".cu");
    writer << boilerplate::MIT1->get_code();

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::cuda);
    re->require(macro::CUDA_SAFE_CALL);
    re->require(declaration::typedef_int);
    re->require(macro::HALF_MAX);
    re->require(header::cublas);

    for (auto& it : re->local_symbol)
    {
        if (it.second->symbol.find("header::") != string::npos)
            writer << it.second->get_code();
    }

    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
        {
            for (auto& sub : it.second->local_symbol)
            {
                writer << sub.second->get_code();
            }
            writer << it.second->get_code();
        }
    writer << "\n";
    if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(ke->kernel))
    {
        if (kernel->require_cudnn_handle())
        {
            writer << "cudnnHandle_t cudnn_handle_0;\n";
        }
        if (kernel->require_cublas_handle())
        {
            writer << "cublasHandle_t cublas_handle_0;\n";
        }
    }
    // special for dropout
    std::unordered_set<std::string> dropout_prefix;
    for (auto& it : re->local_symbol)
    {
        auto symbol_name = it.second->symbol;
        if (symbol_name.find("declaration::dropout_") == string::npos)
            continue;
        auto position = symbol_name.find("dropout");
        dropout_prefix.insert(it.second->symbol.substr(position));
    }

    std::string body_unit = fu->body_unit->get_code();

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << body_unit << "\n";
    writer.block_end();

    // Write Test Calls
    // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
    //{
    //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
    //}

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;
    auto& temp = ke->kernel->m_context->tensors;
    std::unordered_map<size_t, size_t> inplace_map;

    // Support inplace annoation
    if (ke->kernel->m_context->annotations != nullptr)
    {
        auto anno = ke->kernel->m_context->annotations;
        for (auto& pair : anno->get_in_place_oi_pairs())
            if (!pair.destructive)
            {
                inplace_map[pair.output] = pair.input;
                //\todo(wenxh): Support Concat operator's tensor layout;
                if (pair.input_offset != 0)
                {
                    inplace_map.clear();
                    break;
                }
            }
    }

    // Dedupe Tensors
    std::map<std::string, size_t> arg_cnt;
    std::map<std::string, size_t> out_cnt;
    std::map<std::string, size_t> temp_cnt;
    std::vector<std::string> arg_name_dedupe;
    std::vector<std::string> out_name_dedupe;
    std::vector<std::string> temp_name_dedupe;
    for (size_t i = 0; i < arg.size(); i++)
    {
        auto name = arg[i]->get_name();
        if (arg_cnt.find(name) == arg_cnt.end())
        {
            arg_cnt[name] = 1;
            arg_name_dedupe.push_back(name);
        }
        else
        {
            arg_name_dedupe.push_back(name + std::to_string(arg_cnt[name]));
            arg_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < out.size(); i++)
    {
        auto name = out[i]->get_name();
        if (out_cnt.find(name) == out_cnt.end())
        {
            out_cnt[name] = 1;
            out_name_dedupe.push_back(name);
        }
        else
        {
            out_name_dedupe.push_back(name + std::to_string(out_cnt[name]));
            out_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < temp.size(); i++)
    {
        auto name = temp[i]->get_name();
        if (temp_cnt.find(name) == temp_cnt.end())
        {
            temp_cnt[name] = 1;
            temp_name_dedupe.push_back(name);
        }
        else
        {
            temp_name_dedupe.push_back(name + std::to_string(temp_cnt[name]));
            temp_cnt[name] += 1;
        }
    }

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i]->get_element_type().c_type_string() << "* " << arg_name_dedupe[i]
               << "_host, ";
    }
    if (!arg.empty())
    {
        writer << arg.back()->get_element_type().c_type_string() << "* " << arg_name_dedupe.back();
        if (!out.empty())
            writer << "_host, ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i]->get_element_type().c_type_string() << "* " << out_name_dedupe[i]
               << "_host, ";
    }
    if (!out.empty())
    {
        writer << out.back()->get_element_type().c_type_string() << "* " << out_name_dedupe.back()
               << "_host";
    }
    writer << ")\n";

    writer.block_begin();
    {
        if (require_cudnn_handle)
        {
            writer << "CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));\n";
        }
        if (require_cublas_handle)
        {
            writer << "CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));\n";
        }
        for (auto prefix : dropout_prefix)
        {
            writer << prefix << "_init(cudnn_handle_0);\n";
        }

        if (re->local_symbol.count("declaration::num_SMs") > 0)
        {
            writer << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                      "cudaDevAttrMultiProcessorCount, 0));\n";
        }

        auto tensor_declare = [](const shared_ptr<nnfusion::descriptor::Tensor>& t) -> std::string {
            return t->get_element_type().c_type_string() + "* " + t->get_name() + ";\n";
        };

        auto tensor_alloc_cuda = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "cudaMalloc((void**)&" << tensor->get_name() << "," << tensor->size(false) << " * "
              << tensor->get_element_type().size() << ");\n";
            return s.str();
        };

        auto tensor_alloc_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << tensor->get_name() << " = new " << tensor->get_element_type().c_type_string()
              << "[" << tensor->size(false) << "];\n";
            return s.str();
        };

        auto tensor_cpy_h2d = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor,
                                 string from) {
            stringstream s;
            s << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor->get_name() << ", " << from << ", "
              << tensor->size(false) << " * " << tensor->get_element_type().size() << ", "
              << "cudaMemcpyHostToDevice));\n";
            return s.str();
        };

        auto tensor_cpy_d2h = [](string to,
                                 const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "CUDA_SAFE_CALL(cudaMemcpy(" << to << ", " << tensor->get_name() << ", "
              << tensor->size(false) << " * " << tensor->get_element_type().size() << ", "
              << "cudaMemcpyDeviceToHost));\n";
            return s.str();
        };

        auto tensor_free_cuda = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "CUDA_SAFE_CALL(cudaFree(" << tensor->get_name() << "));\n";
            return s.str();
        };

        auto tensor_free_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "delete[] " << tensor->get_name() << ";\n";
            return s.str();
        };

        auto tensor_memset_cuda = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "CUDA_SAFE_CALL(cudaMemset((void*)" << tensor->get_name() << ", "
              << tensor->get_memset_value() << ", " << tensor->size(false) << " * "
              << tensor->get_element_type().size() << "));\n";
            return s.str();
        };

        auto tensor_memset_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "memset(" << tensor->get_name() << tensor->get_memset_value() << ", "
              << tensor->size(false) << " * " << tensor->get_element_type().size() << ");\n";
            return s.str();
        };

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto deduped_name = arg_name_dedupe[i];
            auto& tensor = arg[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                writer << tensor_alloc_cuda(tensor);
                writer << tensor_cpy_h2d(tensor, tensor->get_name() + "_host");
            }
        }

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                if (inplace_map.find(i) == inplace_map.end())
                {
                    writer << tensor_alloc_cuda(tensor);
                    if (tensor->is_memset())
                    {
                        writer << tensor_memset_cuda(tensor);
                    }
                }
                else
                {
                    writer << tensor->get_name() << " = "
                           << arg[inplace_map[i]]->get_name() + ";\n";
                }
            }
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                writer << tensor_alloc_cuda(tensor);
                if (tensor->is_memset())
                {
                    writer << tensor_memset_cuda(tensor);
                }
            }
        }

        writer << "cudaEvent_t start, stop;\n";
        writer << "CUDA_SAFE_CALL(cudaEventCreate(&start));\n";
        writer << "CUDA_SAFE_CALL(cudaEventCreate(&stop));\n";

        fu = ke->kernel->get_or_emit_source(true);

        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times
                   << ") CUDA_SAFE_CALL(cudaEventRecord(start));\n";
            writer << fu->name_unit->get_code() << fu->call_unit->get_code();
        }
        writer.block_end();

        writer << "CUDA_SAFE_CALL(cudaEventRecord(stop));\n";
        writer << "CUDA_SAFE_CALL(cudaEventSynchronize(stop));\n";
        writer << "float milliseconds = 0;\n";
        writer << "CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));\n";

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_cpy_d2h(tensor->get_name() + "_host", tensor);
            }
        }

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto deduped_name = arg_name_dedupe[i];
            auto& tensor = arg[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_free_cuda(tensor);
            }
        }

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                if (inplace_map.find(i) == inplace_map.end())
                {
                    writer << tensor_free_cuda(tensor);
                }
            }
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                if (tensor->get_device_type() != GENERIC_CPU)
                    writer << tensor_free_cuda(tensor);
                else
                    writer << tensor_free_host(tensor);
            }
        }
        for (auto prefix : dropout_prefix)
        {
            writer << prefix << "_free();\n";
        }

        if (require_cudnn_handle)
        {
            writer << "CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));\n";
        }
        if (require_cublas_handle)
        {
            writer << "CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));\n";
        }

        writer << "return milliseconds/" << ke->runtime_times << ";\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "]";
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    return true;
    /*
    // Save the file
    ofstream source_file(writer.symbol);
    source_file << writer.get_code();
    source_file.close();
    return false;
    */
}

bool CudaDefaultRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = filename + ".cu";
    // ofstream source_file(ke->working_dir + "/" + ke->source_code->symbol);
    NNFUSION_LOG(DEBUG) << "complie source file: " << srcname;
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();

    int ret = system(("nvcc\t-lcudnn\t-lcublas\t--compiler-options\t'-fPIC\t "
                      "--shared'\t--cudart\tshared\t-O2\t-gencode="
                      "arch=compute_60,code=compute_60\t-gencode=arch=compute_61,code=compute_61\t-"
                      "std=c++11\t--expt-relaxed-constexpr\t" +
                      srcname + "\t-o\t" + objname)
                         .c_str());

    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

double CudaDefaultRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (codegen(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

CudaDefaultRuntime::Pointer CudaDefaultRuntime::Runtime()
{
    static CudaDefaultRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<CudaDefaultRuntime>();
    return predefined;
}

bool CUPTIRuntime::codegen(const ProfilingContext::Pointer& ke)
{
    if (ke->source_code != nullptr)
        return true;

    // assign async info
    auto async_manager =
        nnfusion::async::AsyncManagerFactory::get_device_stream_async_manager(nullptr, CUDA_GPU);
    auto gnode = ke->kernel->m_context->gnode;
    (*gnode)["Async_info"] = nnfusion::async::AsyncExecutionInfo();
    auto& async_info = (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
    async_info.execution_stream = async_manager->set_stream();
    bool require_cudnn_handle = false;
    bool require_cublas_handle = false;
    if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(ke->kernel))
    {
        if (kernel->require_cudnn_handle())
        {
            async_info.execution_stream->add_binding_symbol("cudnn_handle");
            require_cudnn_handle = true;
        }
        if (kernel->require_cublas_handle())
        {
            async_info.execution_stream->add_binding_symbol("cublas_handle");
            require_cublas_handle = true;
        }
    }

    FunctionUnit_p fu = ke->kernel->get_or_emit_source();
    LanguageUnit writer(fu->name_unit->get_code() + ".cu");
    writer << boilerplate::MIT1->get_code();

    auto re = fu->dep_unit;
    re->require(header::assert);
    re->require(header::stdexcept);
    re->require(header::sstream);
    re->require(header::cuda);
    re->require(header::cupti);
    re->require(header::stdio);
    re->require(macro::CUDA_SAFE_CALL);
    re->require(macro::CUPTI_CALL);
    re->require(declaration::typedef_int);
    re->require(macro::HALF_MAX);
    re->require(header::cublas);

    // Write Dependency
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";

    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            writer << it.second->get_code();
    writer << "\n";
    for (auto& it : re->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
        {
            for (auto& sub : it.second->local_symbol)
            {
                writer << sub.second->get_code();
            }
            writer << it.second->get_code();
        }
    writer << "\n";

    string code = R"(#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

unsigned long long total_kernel_time = 0;
unsigned long long total_kernel_call = 0;

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL)
  {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0)
  {
    do
    {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS)
      {
        switch (record->kind)
        {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
          CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
          total_kernel_time += (unsigned long long)(kernel->end - kernel->start);
          total_kernel_call++;
          break;
        }
        default:
          break;
        }
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else
      {
        CUPTI_CALL(status);
      }
    } while (1);
  }

  free(buffer);
}
    )";
    writer << code << "\n";
    if (auto kernel = std::dynamic_pointer_cast<CudaLibEmitter>(ke->kernel))
    {
        if (kernel->require_cudnn_handle())
        {
            writer << "cudnnHandle_t cudnn_handle_0;\n";
        }
        if (kernel->require_cublas_handle())
        {
            writer << "cublasHandle_t cublas_handle_0;\n";
        }
    }

    std::string body_unit = fu->body_unit->get_code();

    // Write function definition
    writer << fu->comment_unit->get_code();
    writer << fu->get_specialized_signature() << "\n";
    writer.block_begin();
    writer << body_unit << "\n";
    writer.block_end();

    // Write Test Calls
    // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
    //{
    //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
    //}

    auto& arg = ke->kernel->m_context->inputs;
    auto& out = ke->kernel->m_context->outputs;
    auto& temp = ke->kernel->m_context->tensors;
    std::unordered_map<size_t, size_t> inplace_map;

    // Support inplace annoation
    if (ke->kernel->m_context->annotations != nullptr)
    {
        auto anno = ke->kernel->m_context->annotations;
        for (auto& pair : anno->get_in_place_oi_pairs())
            if (!pair.destructive)
            {
                inplace_map[pair.output] = pair.input;
                //\todo(wenxh): Support Concat operator's tensor layout;
                if (pair.input_offset != 0)
                {
                    inplace_map.clear();
                    break;
                }
            }
    }

    // Dedupe Tensors
    std::map<std::string, size_t> arg_cnt;
    std::map<std::string, size_t> out_cnt;
    std::map<std::string, size_t> temp_cnt;
    std::vector<std::string> arg_name_dedupe;
    std::vector<std::string> out_name_dedupe;
    std::vector<std::string> temp_name_dedupe;
    for (size_t i = 0; i < arg.size(); i++)
    {
        auto name = arg[i]->get_name();
        if (arg_cnt.find(name) == arg_cnt.end())
        {
            arg_cnt[name] = 1;
            arg_name_dedupe.push_back(name);
        }
        else
        {
            arg_name_dedupe.push_back(name + std::to_string(arg_cnt[name]));
            arg_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < out.size(); i++)
    {
        auto name = out[i]->get_name();
        if (out_cnt.find(name) == out_cnt.end())
        {
            out_cnt[name] = 1;
            out_name_dedupe.push_back(name);
        }
        else
        {
            out_name_dedupe.push_back(name + std::to_string(out_cnt[name]));
            out_cnt[name] += 1;
        }
    }
    for (size_t i = 0; i < temp.size(); i++)
    {
        auto name = temp[i]->get_name();
        if (temp_cnt.find(name) == temp_cnt.end())
        {
            temp_cnt[name] = 1;
            temp_name_dedupe.push_back(name);
        }
        else
        {
            temp_name_dedupe.push_back(name + std::to_string(temp_cnt[name]));
            temp_cnt[name] += 1;
        }
    }

    writer << "extern \"C\" double " << fu->name_unit->get_code() << "_host(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i]->get_element_type().c_type_string() << "* " << arg_name_dedupe[i]
               << "_host, ";
    }
    if (!arg.empty())
    {
        writer << arg.back()->get_element_type().c_type_string() << "* " << arg_name_dedupe.back();
        if (!out.empty())
            writer << "_host, ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i]->get_element_type().c_type_string() << "* " << out_name_dedupe[i]
               << "_host, ";
    }
    if (!out.empty())
    {
        writer << out.back()->get_element_type().c_type_string() << "* " << out_name_dedupe.back()
               << "_host";
    }
    writer << ")\n";

    writer.block_begin();
    {
        if (require_cudnn_handle)
        {
            writer << "CUDNN_SAFE_CALL(cudnnCreate(&cudnn_handle_0));\n";
        }
        if (require_cublas_handle)
        {
            writer << "CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_0));\n";
        }

        if (re->local_symbol.count("declaration::num_SMs") > 0)
        {
            writer << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
                      "cudaDevAttrMultiProcessorCount, 0));\n";
        }

        auto tensor_declare = [](const shared_ptr<nnfusion::descriptor::Tensor>& t) -> std::string {
            return t->get_element_type().c_type_string() + "* " + t->get_name() + ";\n";
        };

        auto tensor_alloc_cuda = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "cudaMalloc((void**)&" << tensor->get_name() << "," << tensor->size(false) << " * "
              << tensor->get_element_type().size() << ");\n";
            return s.str();
        };

        auto tensor_alloc_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << tensor->get_name() << " = new " << tensor->get_element_type().c_type_string()
              << "[" << tensor->size(false) << "];\n";
            return s.str();
        };

        auto tensor_cpy_h2d = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor,
                                 string from) {
            stringstream s;
            s << "cudaMemcpy(" << tensor->get_name() << ", " << from << ", " << tensor->size(false)
              << " * " << tensor->get_element_type().size() << ", "
              << "cudaMemcpyHostToDevice);\n";
            return s.str();
        };

        auto tensor_cpy_d2h = [](string to,
                                 const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "cudaMemcpy(" << to << ", " << tensor->get_name() << ", " << tensor->size(false)
              << " * " << tensor->get_element_type().size() << ", "
              << "cudaMemcpyDeviceToHost);\n";
            return s.str();
        };

        auto tensor_free_cuda = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "CUDA_SAFE_CALL(cudaFree(" << tensor->get_name() << "));\n";
            return s.str();
        };

        auto tensor_free_host = [](const shared_ptr<nnfusion::descriptor::Tensor>& tensor) {
            stringstream s;
            s << "delete[] " << tensor->get_name() << ";\n";
            return s.str();
        };

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto deduped_name = arg_name_dedupe[i];
            auto& tensor = arg[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                writer << tensor_alloc_cuda(tensor);
                writer << tensor_cpy_h2d(tensor, tensor->get_name() + "_host");
            }
        }

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                if (inplace_map.find(i) == inplace_map.end())
                {
                    writer << tensor_alloc_cuda(tensor);
                }
                else
                {
                    writer << tensor->get_name() << " = "
                           << arg[inplace_map[i]]->get_name() + ";\n";
                }
            }
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_declare(tensor);
                writer << tensor_alloc_cuda(tensor);
            }
        }

        writer << "cudaEvent_t start, stop;\n";
        writer << "cudaEventCreate(&start);\n";
        writer << "cudaEventCreate(&stop);\n";

        fu = ke->kernel->get_or_emit_source(true);

        writer << "for(int i=0; i < " << ke->warmup_times + ke->runtime_times << "; i++)\n";
        writer.block_begin();
        {
            writer << "if(i == " << ke->warmup_times << ") {\n"
                   << "  cudaEventRecord(start);\n"
                   << "  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));\n"
                   << "  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, "
                      "bufferCompleted));\n"
                   << "}\n";
            writer << fu->name_unit->get_code() << fu->call_unit->get_code();
        }
        writer.block_end();

        writer << "cudaEventRecord(stop);\n";
        writer << "cudaEventSynchronize(stop);\n";
        writer << "cuptiActivityFlushAll(0);\n";
        writer << "float milliseconds = 0;\n";
        writer << "cudaEventElapsedTime(&milliseconds, start, stop);\n";

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_cpy_d2h(tensor->get_name() + "_host", tensor);
            }
        }

        for (size_t i = 0; i < arg.size(); i++)
        {
            auto deduped_name = arg_name_dedupe[i];
            auto& tensor = arg[i];
            if (deduped_name == tensor->get_name())
            {
                writer << tensor_free_cuda(tensor);
            }
        }

        for (size_t i = 0; i < out.size(); i++)
        {
            auto deduped_name = out_name_dedupe[i];
            auto& tensor = out[i];
            if (deduped_name == tensor->get_name())
            {
                if (inplace_map.find(i) == inplace_map.end())
                {
                    writer << tensor_free_cuda(tensor);
                }
            }
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            auto deduped_name = temp_name_dedupe[i];
            auto& tensor = temp[i];
            if (deduped_name == tensor->get_name())
            {
                if (tensor->get_device_type() != GENERIC_CPU)
                    writer << tensor_free_cuda(tensor);
                else
                    writer << tensor_free_host(tensor);
            }
        }

        if (require_cudnn_handle)
        {
            writer << "CUDNN_SAFE_CALL(cudnnDestroy(cudnn_handle_0));\n";
        }
        if (require_cublas_handle)
        {
            writer << "CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_0));\n";
        }

        //writer << "return milliseconds/" << ke->runtime_times << ";\n";
        writer << "return (double)total_kernel_time / total_kernel_call / 1000;\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" double " << fu->name_unit->get_code()
           << "_entry(void** args, void** outputs)";

    writer.block_begin();
    {
        writer << "return " << fu->name_unit->get_code() << "_host(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (i < arg.size() ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i]->get_element_type().c_type_string()
                              : (i - arg.size() < out.size()
                                     ? out[i - arg.size()]->get_element_type().c_type_string()
                                     : "");
            writer << "(" << type << "*)" << (out.size() == 0 ? "args" : "outputs") << "["
                   << i - (i >= arg.size() ? arg.size() : 0) << "]";
        }
        writer << ");\n";
    }
    writer.block_end();

    ke->source_code = make_shared<LanguageUnit>(move(writer));
    return true;
    /*
    // Save the file
    ofstream source_file(writer.symbol);
    source_file << writer.get_code();
    source_file.close();
    return false;
    */
}

bool CUPTIRuntime::compile(const ProfilingContext::Pointer& ke)
{
    if (ke->entry_point != nullptr)
        return true;
    string filename = string(tmpnam(nullptr));
    string objname = filename + DLIB_SUFFIX;
    string srcname = filename + ".cu";
    // ofstream source_file(ke->working_dir + "/" + ke->source_code->symbol);
    ofstream source_file(srcname);
    source_file << ke->source_code->get_code();
    source_file.close();
    int ret =
        system(("nvcc\t-lcudnn\t-lcublas\t-lcupti\t--compiler-options\t'-fPIC\t "
                "-I/usr/local/cuda/extras/CUPTI/include\t-L/usr/local/cuda/extras/CUPTI/lib64\t"
                "--shared'\t--cudart\tshared\t-O2\t-gencode="
                "arch=compute_60,code=compute_60\t-gencode=arch=compute_61,code=compute_61\t"
                "-gencode=arch=compute_70,code=compute_70\t-gencode=arch=compute_75,code=compute_"
                "75\t-std=c++11\t--expt-relaxed-constexpr\t" +
                srcname + "\t-o\t" + objname)
                   .c_str());
    if (ret != 0)
        return false;
    if (!file_exsits(objname))
        return false;
    auto obj = get_library_handle(objname);
    auto entry = get_funcion_pointer(
        ke->kernel->get_or_emit_source()->name_unit->get_code() + "_entry", obj);
    if (entry == nullptr)
        return false;
    ke->entry_point = (double (*)(void**, void**))entry;
    return true;
}

double CUPTIRuntime::invoke(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (codegen(ke) == false)
        return -1.0;
    if (compile(ke) == false)
        return -1.0;
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}

CUPTIRuntime::Pointer CUPTIRuntime::Runtime()
{
    static CUPTIRuntime::Pointer predefined = nullptr;
    if (predefined == nullptr)
        predefined = make_shared<CUPTIRuntime>();
    return predefined;
}
