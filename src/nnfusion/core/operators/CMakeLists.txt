# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


file(GLOB generic_op_define generic_op/generic_op_define/*.cpp)

set(SRC
    op.cpp
    util/arithmetic_reduction.cpp
    util/elementwise_arithmetic.cpp
    util/binary_elementwise_comparison.cpp
    util/binary_elementwise_logical.cpp
    util/index_reduction.cpp
    util/tensor_op.cpp
    util/validation_util.cpp
    generic_op/op_registration.cpp
    op_define/abs.cpp
    op_define/acos.cpp
    op_define/add.cpp
    op_define/allreduce.cpp
    op_define/and.cpp
    op_define/argmax.cpp
    op_define/argmin.cpp
    op_define/asin.cpp
    op_define/atan.cpp
    op_define/avg_pool.cpp
    op_define/batch_norm.cpp
    op_define/broadcast.cpp
    op_define/ceiling.cpp
    op_define/concat.cpp
    op_define/constant.cpp
    op_define/convert.cpp
    op_define/convolution.cpp
    op_define/cos.cpp
    op_define/cosh.cpp
    op_define/divide.cpp
    op_define/dot.cpp
    op_define/equal.cpp
    op_define/erf.cpp
    op_define/exp.cpp
    op_define/floor.cpp
    op_define/gelu.cpp
    op_define/greater_eq.cpp
    op_define/greater.cpp
    op_define/if.cpp
    op_define/less_eq.cpp
    op_define/less.cpp
    op_define/log.cpp
    op_define/loop.cpp
    op_define/lrn.cpp
    op_define/max_pool.cpp
    op_define/max.cpp
    op_define/maximum.cpp
    op_define/min.cpp
    op_define/minimum.cpp
    op_define/multiply.cpp
    op_define/negative.cpp
    op_define/not_equal.cpp
    op_define/not.cpp
    op_define/or.cpp
    op_define/pad.cpp
    op_define/parameter.cpp
    op_define/power.cpp
    op_define/product.cpp
    op_define/reduce_window.cpp
    op_define/reduce.cpp
    op_define/relu.cpp
    op_define/replace_slice.cpp
    op_define/reshape.cpp
    op_define/result.cpp
    op_define/reverse_sequence.cpp
    op_define/reverse.cpp
    op_define/rsqrt.cpp
    op_define/select_and_scatter.cpp
    op_define/select.cpp
    op_define/sigmoid.cpp
    op_define/sign.cpp
    op_define/sin.cpp
    op_define/sinh.cpp
    op_define/slice.cpp
    op_define/softmax.cpp
    op_define/sqrt.cpp
    op_define/stop_gradient.cpp
    op_define/subtract.cpp
    op_define/sum.cpp
    op_define/square.cpp
    op_define/tan.cpp
    op_define/tanh.cpp
    op_define/topk.cpp
    op_define/variable.cpp
    op_define/fused.cpp
    ${generic_op_define}
)

add_library(nnfusion_operators SHARED ${SRC})
target_include_directories(nnfusion_operators SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(nnfusion_operators nnfusion_common contrib_custom_operators)
target_compile_options(nnfusion_operators PRIVATE "-fPIC")

install(TARGETS nnfusion_operators
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
)