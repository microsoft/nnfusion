nohup sh test_basic.sh > log_basic 2>&1 &

nohup sh test_conv_fused_implicit_gemm.sh > log_conv_fused_implicit_gemm 2>&1 &

nohup sh test_depthwise_conv_fused.sh > log_depthwise_conv_fused 2>&1 &
