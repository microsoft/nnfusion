import subprocess
import os

Tasks = {
    ("bert", 1, "fp32"): ["./bert_gemm 1 128 12 64 0 0 1 1", "./bert_example 1 12 128 12 64 0 0"],
    ("bert", 1, "fp16"): ["./bert_gemm 1 128 12 64 1 0 1 1", "./bert_example 1 12 128 12 64 1 0"],
    ("bert", 64, "fp32"): ["./bert_gemm 64 128 12 64 0 0 1 1", "./bert_example 64 12 128 12 64 0 0"],
    ("bert", 64, "fp16"): ["./bert_gemm 64 128 12 64 1 0 1 1", "./bert_example 64 12 128 12 64 1 0"],

    ("vit", 1, "fp32"): ["./vit_gemm 1 224 32 384 6 1 0 0", "./vit_example 1 224 32 384 6 12 1 0"],
    ("vit", 1, "fp16"): ["./vit_gemm 1 224 32 384 6 1 1 0", "./vit_example 1 224 32 384 6 12 1 1"],
    ("vit", 64, "fp32"): ["./vit_gemm 64 224 32 384 6 1 0 0", "./vit_example 64 224 32 384 6 12 1 0"],
    ("vit", 64, "fp16"): ["./vit_gemm 64 224 32 384 6 1 1 0", "./vit_example 64 224 32 384 6 12 1 1"],

    ("swin", 1, "fp32"): ["./swin_gemm 1 224 7 3 32 0 0", "./swin_example 1 0 0 7 224 1"],
    ("swin", 1, "fp16"): ["./swin_gemm 1 224 7 3 32 1 0", "./swin_example 1 1 0 7 224 1"],
    ("swin", 64, "fp32"): ["./swin_gemm 64 224 7 3 32 0 0", "./swin_example 1 0 0 7 224 64"],
    ("swin", 64, "fp16"): ["./swin_gemm 64 224 7 3 32 1 0", "./swin_example 1 1 0 7 224 64"],
}

if __name__ == "__main__":
    prefix = "/root/FasterTransformer/build/bin"
    cur_dir = os.path.abspath(".")
    os.chdir(prefix)
    for key in Tasks:
        print("Running", key)
        for command in Tasks[key]:
            ret = subprocess.run(command.split(), capture_output=True)
            for line in ret.stdout.decode().split("\n"):
                if "FT-CPP-time" in line:
                    print(line)
    os.chdir(cur_dir)
