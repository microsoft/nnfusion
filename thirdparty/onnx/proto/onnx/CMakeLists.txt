file(GLOB protos *.proto)

protobuf_generate_cpp(srcs hdrs ${protos})
add_library(onnx_proto STATIC 
        ${srcs}
        ${hdrs}
)