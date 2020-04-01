// Microsoft (c) 2019, NNFusion Team
#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Constant : public KernelEmitter
            {
            public:
                Constant(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda")
                {
                    op = static_pointer_cast<nnfusion::op::Constant>(ctx->gnode->get_op_ptr());
                    CHECK_NOT_NULLPTR(op) << "Node type is not Constant.";

                    folder = "./Constant/";

                    std::stringstream tag;
                    tag << "load_" << const_name;
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    nnfusion::codegen::create_folder(folder);
                    const_name = m_context->outputs[0]->get_name();
                    ofstream bin_file(folder + const_name + ".bin", ios::out | ios::binary);
                    bin_file.write((const char*)op->get_data_ptr(), op->get_data_size());
                    bin_file.close();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;
                    writer << "std::ifstream bin_file(\"" << folder + const_name
                           << ".bin\" , std::ios::in | std::ios::binary);\n"
                           // << "cudaMalloc((void**)out, " << op->get_data_size() << ");\n"
                           << "char* tmp_mem = new char[" << op->get_data_size() << "];\n"
                           << "bin_file.read(tmp_mem, " << op->get_data_size() << ");\n"
                           << "cudaMemcpyAsync(output0, tmp_mem, " << op->get_data_size()
                           << ", cudaMemcpyHostToDevice, stream);\n"
                           << "bin_file.close();\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::fstream);
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<nnfusion::op::Constant> op;
                string folder;
                string const_name;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Constant",                                //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT), //attrs
                        cuda::Constant)                            // constructor