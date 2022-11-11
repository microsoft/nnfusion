#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ArgMax: public BlockCudaEmitter
            {
                public:
                    ArgMax(shared_ptr<KernelContext> ctx): BlockCudaEmitter(ctx)
                    {
                        m_op = dynamic_pointer_cast<nnfusion::op::ArgMax>(ctx->gnode->get_op_ptr());
                        NNFUSION_CHECK_NOT_NULLPTR(m_op) << "Node type is not ArgMax.";
                        m_input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                        m_output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                        m_blockDim = dim3(32, 1, 1); // each block processes an output element
                        num_local_thread_sync = 5;
                        while (m_blockDim.x < m_input_shape[m_op->get_reduction_axis()] && m_blockDim.x < 256)
                        {
                            m_blockDim.x *= 2;
                            num_local_thread_sync ++;
                        } // max m_blockDim is 256 
                        m_gridDim = dim3(shape_size(m_output_shape), 1, 1);
                    }

                    LanguageUnit_p emit_function_body() {
                        LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                        auto& lu = *_lu;
                        int in_reduce_size = 1;
                        for (int i = m_op->get_reduction_axis() + 1; i < m_input_shape.size(); i++) {
                            in_reduce_size *= m_input_shape[i];
                        }
                        int reduce_size = m_input_shape[m_op->get_reduction_axis()];
                        std::string T_MIN;
                        nnfusion::element::Type T = m_context->inputs[0]->get_element_type();
                        if (T == nnfusion::element::f32) {
                            T_MIN = "-FLT_MAX";
                        } else if (T == nnfusion::element::f64) {
                            T_MIN = "-DBL_MAX";
                        } else if (T == nnfusion::element::i32) {
                            T_MIN = "INT_MIN";
                        } else if (T == nnfusion::element::i64) {
                            T_MIN = "LONG_MIN";
                        } else {
                            NNFUSION_CHECK_FAIL() << "ArgMax not support type " << T.c_type_string();
                        }

                        auto code = nnfusion::op::create_code_from_template(
                            R"(
int in_reduce_size = @IN_REDUCE_SIZE@;
int reduce_size = @REDUCE_SIZE@;
int out_id = blockIdx.x / in_reduce_size;
int in_id = blockIdx.x % in_reduce_size;
int bias = out_id * reduce_size * in_reduce_size + in_id;
int max_id = -1;
@T@ max_value = @T_MIN@;
for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
    @T@ value = input0[bias + i * in_reduce_size];
    if (value > max_value) {
        max_value = value;
        max_id = i;
    }
}
    )",                     {
                                {"IN_REDUCE_SIZE", in_reduce_size},
                                {"REDUCE_SIZE", reduce_size},
                                {"T", T.c_type_string()},
                                {"T_MIN", T_MIN},
                        });
                        lu << code;
                        emit_alloc_shared(lu, "shared_max_value", T.c_type_string(), m_blockDim.x);
                        emit_alloc_shared(lu, "shared_max_id", m_op->get_index_element_type().c_type_string(), m_blockDim.x);

                        code = nnfusion::op::create_code_from_template(
                            R"(
shared_max_value[threadIdx.x] = max_value;
shared_max_id[threadIdx.x] = max_id;
__syncthreads();
# pragma unroll
for (int i = @BLOCK_DIM@ / 2; i > 0; i /= 2) {
    if (threadIdx.x < i) {
        if (shared_max_value[threadIdx.x] < shared_max_value[threadIdx.x + i]) {
            shared_max_value[threadIdx.x] = shared_max_value[threadIdx.x + i];
            shared_max_id[threadIdx.x] = shared_max_id[threadIdx.x + i];
        }
    }
    __syncthreads();
}
if (threadIdx.x == 0) {
    output0[out_id * in_reduce_size + in_id] = shared_max_id[0];
}
)",                     {
                                {"BLOCK_DIM", m_blockDim.x}
                        });
                        lu << code;
                        return _lu;
                    }
                    LanguageUnit_p emit_dependency() {
                        LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                        _lu->require(header::cuda);
                        _lu->require(header::FLOAT);
                        return _lu;
                    }
                    void set_launch_config() {
                        return;
                    }
                private:
                    shared_ptr<nnfusion::op::ArgMax> m_op;
                    nnfusion::Shape m_input_shape, m_output_shape;
                    nnfusion::AxisSet m_reduce_axis;
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("ArgMax",                                      //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32), //attrs
                        cuda::ArgMax)                                 // constructor
