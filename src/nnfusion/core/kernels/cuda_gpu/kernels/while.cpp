#include "while.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/while.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"
#include <set>
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_int32(floop_copy_blockdim);
DECLARE_bool(floop_in_c);
DECLARE_bool(ffast_barrier);
DECLARE_int32(ffused_max_grid);

cuda::While::While(shared_ptr<KernelContext> ctx)
    : Loop(ctx, 256, 1)
{
    std::stringstream tag;
    tag << "_WhileOP";
    custom_tag = tag.str();
    m_cond_offset = reserved_memory_start;
}

LanguageUnit_p cuda::While::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    if (FLAGS_floop_in_c) {
        // NNFUSION_LOG(INFO) << "not implemented"; exit(1);
        lu << "char cond_host;\n";
        lu << "static char* cond = nullptr;\n";
        lu << "if (cond == nullptr) CUDA_SAFE_CALL(cudaMalloc(&cond, sizeof(char)));\n";
        lu << "CUDA_SAFE_CALL(cudaMemcpy(&cond_host, (char*)input0, sizeof(char), cudaMemcpyDeviceToHost));\n";
        // lu << "if (!cond_host)";
        // lu.block_begin();
        int grid_dim = 0;
        for (int i = 0; i < m_context->outputs.size(); i++) {
            size_t tensor_size = shape_size(m_context->outputs[i]->get_shape());
            grid_dim += ceil_div(tensor_size, (size_t) FLAGS_floop_copy_blockdim);
        }
        std::string copy_func_name = get_function_name() + "_copy_kernel";
        vector<string> params_with_type;
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "input" << i + 1;
            params_with_type.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++)
        {
            stringstream ss;
            ss << "output" << i;
            params_with_type.push_back(ss.str());
        }
        lu << copy_func_name << "<<<" << "dim3(" << grid_dim << ", 1, 1), dim3(" << FLAGS_floop_copy_blockdim << ", 1, 1)" << ">>>(" << join(params_with_type, ", ") << ");\n";
        // lu.block_end();
        for (int i = 0; i < m_context->outputs.size(); i++)
            lu << "input" << i + 1 << " = output" << i << ";\n";
        lu << "while (cond_host)";
        lu.block_begin();
        generate_subgraph_code(_lu, false);
        lu << "CUDA_SAFE_CALL(cudaMemcpy(&cond_host, (char*)cond, sizeof(char), cudaMemcpyDeviceToHost));\n";
        lu.block_end();
    } else {
        allocate_shared_memory(_lu);
        lu << "int tid=threadIdx.x + blockIdx.x * blockDim.x;\n";
        for (int i = 0; i < m_context->outputs.size(); i++)
        {
            size_t tensor_size = m_context->outputs[i]->size(false);
            size_t num_threads = m_blockDim.x * m_gridDim.x;
            lu << "for (int64_t i=tid; i<" << tensor_size << "; i+=" << num_threads << ")";
            lu << " output" << i << "[i] = input" << i + 1 << "[i];\n";
        }
        for (int i = 0; i < m_context->outputs.size(); i++)
            lu << "input" << i + 1  << " = output" << i << ";\n";
        lu << "char* cond = (char*)(input" + std::to_string(m_context->inputs.size() - 1) + "+" + std::to_string(m_cond_offset) + ");\n";
        lu << "if (blockIdx.x == 0 && threadIdx.x == 0) { cond[0] = input0[0]; }\n";
        lu << "Barrier();\n";
        lu << "while (*cond)";
        lu.block_begin();
        generate_subgraph_code(_lu, true);
        lu.block_end();
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER("While",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::While)

