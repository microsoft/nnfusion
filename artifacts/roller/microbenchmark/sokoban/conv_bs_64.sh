# resnet
python3 -u tune_conv.py 64 128 28 28 128 3 3 1 1 1 2>&1 |tee our_conv_64_128_28_28_128_3_3_1_1_1.log
python3 -u tune_conv.py 64 256 14 14 256 3 3 1 1 1 2>&1 |tee our_conv_64_256_14_14_256_3_3_1_1_1.log
python3 -u tune_conv.py 64 512 7 7 512 3 3 1 1 1 2>&1 |tee our_conv_64_512_7_7_512_3_3_1_1_1.log
python3 -u tune_conv.py 64 64 56 56 64 3 3 1 1 1 2>&1 |tee our_conv_64_64_56_56_64_3_3_1_1_1.log
python3 -u tune_conv.py 64 64 56 56 256 1 1 1 1 0 2>&1 |tee our_conv_64_64_56_56_256_1_1_1_1_0.log
python3 -u tune_conv.py 64 128 28 28 512 1 1 1 1 0 2>&1 |tee our_conv_64_128_28_28_512_1_1_1_1_0.log
python3 -u tune_conv.py 64 512 28 28 128 1 1 1 1 0 2>&1 |tee our_conv_64_512_28_28_128_1_1_1_1_0.log
python3 -u tune_conv.py 64 512 7 7 2048 1 1 1 1 0 2>&1 |tee our_conv_64_512_7_7_2048_1_1_1_1_0.log
python3 -u tune_conv.py 64 256 14 14 1024 1 1 1 1 0 2>&1 |tee our_conv_64_256_14_14_1024_1_1_1_1_0.log
python3 -u tune_conv.py 64 1024 14 14 256 1 1 1 1 0 2>&1 |tee our_conv_64_1024_14_14_256_1_1_1_1_0.log
python3 -u tune_conv.py 64 2048 7 7 512 1 1 1 1 0 2>&1 |tee our_conv_64_2048_7_7_512_1_1_1_1_0.log
python3 -u tune_conv.py 64 3 230 230 64 7 7 2 1 0 2>&1 |tee our_conv_64_3_230_230_64_7_7_2_1_0.log
python3 -u tune_conv.py 64 64 56 56 64 1 1 1 1 0 2>&1 |tee our_conv_64_64_56_56_64_1_1_1_1_0.log
python3 -u tune_conv.py 64 256 56 56 64 1 1 1 1 0 2>&1 |tee our_conv_64_256_56_56_64_1_1_1_1_0.log
python3 -u tune_conv.py 64 256 56 56 512 1 1 2 1 0 2>&1 |tee our_conv_64_256_56_56_512_1_1_2_1_0.log
python3 -u tune_conv.py 64 256 56 56 128 1 1 2 1 0 2>&1 |tee our_conv_64_256_56_56_128_1_1_2_1_0.log
python3 -u tune_conv.py 64 512 28 28 1024 1 1 2 1 0 2>&1 |tee our_conv_64_512_28_28_1024_1_1_2_1_0.log
python3 -u tune_conv.py 64 512 28 28 256 1 1 2 1 0 2>&1 |tee our_conv_64_512_28_28_256_1_1_2_1_0.log
python3 -u tune_conv.py 64 1024 14 14 2048 1 1 2 1 0 2>&1 |tee our_conv_64_1024_14_14_2048_1_1_2_1_0.log
python3 -u tune_conv.py 64 1024 14 14 512 1 1 2 1 0 2>&1 |tee our_conv_64_1024_14_14_512_1_1_2_1_0.log

# nasnet imagenet
python3 -u tune_conv.py 64 3 224 224 32 3 3 2 1 0 2>&1 |tee our_conv_64_3_224_224_32_3_3_2_1_0.log
python3 -u tune_conv.py 64 32 111 111 11 1 1 1 1 0 2>&1 |tee our_conv_64_32_111_111_11_1_1_1_1_0.log
python3 -u tune_conv.py 64 11 56 56 11 1 1 1 1 0 2>&1 |tee our_conv_64_11_56_56_11_1_1_1_1_0.log
python3 -u tune_conv.py 64 32 56 56 11 1 1 1 1 0 2>&1 |tee our_conv_64_32_56_56_11_1_1_1_1_0.log
python3 -u tune_conv.py 64 44 56 56 22 1 1 1 1 0 2>&1 |tee our_conv_64_44_56_56_22_1_1_1_1_0.log
python3 -u tune_conv.py 64 22 28 28 22 1 1 1 1 0 2>&1 |tee our_conv_64_22_28_28_22_1_1_1_1_0.log
python3 -u tune_conv.py 64 44 28 28 22 1 1 1 1 0 2>&1 |tee our_conv_64_44_28_28_22_1_1_1_1_0.log
python3 -u tune_conv.py 64 88 28 28 44 1 1 1 1 0 2>&1 |tee our_conv_64_88_28_28_44_1_1_1_1_0.log
python3 -u tune_conv.py 64 44 28 28 44 1 1 1 1 0 2>&1 |tee our_conv_64_44_28_28_44_1_1_1_1_0.log
python3 -u tune_conv.py 64 264 28 28 44 1 1 1 1 0 2>&1 |tee our_conv_64_264_28_28_44_1_1_1_1_0.log
python3 -u tune_conv.py 64 264 28 28 88 1 1 1 1 0 2>&1 |tee our_conv_64_264_28_28_88_1_1_1_1_0.log
python3 -u tune_conv.py 64 88 14 14 88 1 1 1 1 0 2>&1 |tee our_conv_64_88_14_14_88_1_1_1_1_0.log
python3 -u tune_conv.py 64 264 14 14 44 1 1 1 1 0 2>&1 |tee our_conv_64_264_14_14_44_1_1_1_1_0.log
python3 -u tune_conv.py 64 352 14 14 88 1 1 1 1 0 2>&1 |tee our_conv_64_352_14_14_88_1_1_1_1_0.log
python3 -u tune_conv.py 64 528 14 14 88 1 1 1 1 0 2>&1 |tee our_conv_64_528_14_14_88_1_1_1_1_0.log
python3 -u tune_conv.py 64 528 14 14 176 1 1 1 1 0 2>&1 |tee our_conv_64_528_14_14_176_1_1_1_1_0.log
python3 -u tune_conv.py 64 176 7 7 176 1 1 1 1 0 2>&1 |tee our_conv_64_176_7_7_176_1_1_1_1_0.log
python3 -u tune_conv.py 64 528 7 7 88 1 1 1 1 0 2>&1 |tee our_conv_64_528_7_7_88_1_1_1_1_0.log
python3 -u tune_conv.py 64 704 7 7 176 1 1 1 1 0 2>&1 |tee our_conv_64_704_7_7_176_1_1_1_1_0.log
python3 -u tune_conv.py 64 1056 7 7 176 1 1 1 1 0 2>&1 |tee our_conv_64_1056_7_7_176_1_1_1_1_0.log


# nasnet large
python3 -u tune_conv.py 64 3 331 331 96 3 3 2 1 0 2>&1 |tee our_conv_64_3_331_331_96_3_3_2_1_0.log
python3 -u tune_conv.py 64 96 165 165 42 1 1 1 1 0 2>&1 |tee our_conv_64_96_165_165_42_1_1_1_1_0.log
python3 -u tune_conv.py 64 96 83 83 42 1 1 1 1 0 2>&1 |tee our_conv_64_96_83_83_42_1_1_1_1_0.log
python3 -u tune_conv.py 64 42 83 83 42 1 1 1 1 0 2>&1 |tee our_conv_64_42_83_83_42_1_1_1_1_0.log
python3 -u tune_conv.py 64 84 42 42 84 1 1 1 1 0 2>&1 |tee our_conv_64_84_42_42_84_1_1_1_1_0.log
python3 -u tune_conv.py 64 168 83 83 84 1 1 1 1 0 2>&1 |tee our_conv_64_168_83_83_84_1_1_1_1_0.log
python3 -u tune_conv.py 64 168 42 42 84 1 1 1 1 0 2>&1 |tee our_conv_64_168_42_42_84_1_1_1_1_0.log
python3 -u tune_conv.py 64 168 42 42 168 1 1 1 1 0 2>&1 |tee our_conv_64_168_42_42_168_1_1_1_1_0.log
python3 -u tune_conv.py 64 336 42 42 168 1 1 1 1 0 2>&1 |tee our_conv_64_336_42_42_168_1_1_1_1_0.log
python3 -u tune_conv.py 64 1008 42 42 168 1 1 1 1 0 2>&1 |tee our_conv_64_1008_42_42_168_1_1_1_1_0.log
python3 -u tune_conv.py 64 1008 42 42 336 1 1 1 1 0 2>&1 |tee our_conv_64_1008_42_42_336_1_1_1_1_0.log
python3 -u tune_conv.py 64 1008 21 21 168 1 1 1 1 0 2>&1 |tee our_conv_64_1008_21_21_168_1_1_1_1_0.log
python3 -u tune_conv.py 64 336 21 21 336 1 1 1 1 0 2>&1 |tee our_conv_64_336_21_21_336_1_1_1_1_0.log
python3 -u tune_conv.py 64 1344 21 21 336 1 1 1 1 0 2>&1 |tee our_conv_64_1344_21_21_336_1_1_1_1_0.log
python3 -u tune_conv.py 64 2016 21 21 336 1 1 1 1 0 2>&1 |tee our_conv_64_2016_21_21_336_1_1_1_1_0.log
python3 -u tune_conv.py 64 2016 21 21 672 1 1 1 1 0 2>&1 |tee our_conv_64_2016_21_21_672_1_1_1_1_0.log
python3 -u tune_conv.py 64 2016 11 11 336 1 1 1 1 0 2>&1 |tee our_conv_64_2016_11_11_336_1_1_1_1_0.log
python3 -u tune_conv.py 64 672 11 11 672 1 1 1 1 0 2>&1 |tee our_conv_64_672_11_11_672_1_1_1_1_0.log
python3 -u tune_conv.py 64 2688 11 11 672 1 1 1 1 0 2>&1 |tee our_conv_64_2688_11_11_672_1_1_1_1_0.log
python3 -u tune_conv.py 64 4032 11 11 672 1 1 1 1 0 2>&1 |tee our_conv_64_4032_11_11_672_1_1_1_1_0.log

