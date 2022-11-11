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
    : Loop(ctx)
{
    std::stringstream tag;
    tag << "_WhileOP";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::While::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    if (FLAGS_floop_in_c) {
        NNFUSION_LOG(INFO) << "not implemented"; exit(1);
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
        lu << "Barrier();\n";
        lu << "char cond = *input0;\n";
        lu << "while (cond)";
        lu.block_begin();
        generate_subgraph_code(_lu, true);
        lu.block_end();
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER("While",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::While)

