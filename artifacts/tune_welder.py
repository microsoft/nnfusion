import subprocess
import os

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--arch', type=str, default="V100")
    parser.add_argument("--no_tc", action="store_true", default=False)
    parser.add_argument("--welder_base", action="store_true", default=False)
    parser.add_argument("--welder_none", action="store_true", default=False)
    parser.add_argument("--skip_dot", action="store_true", default=False)
    args = parser.parse_args()

    os.chdir(args.prefix)
    command1 = ["nnfusion", "model.onnx", "-f", "onnx", "-ftune_output_file=model.json"]
    command2 = ["python", "-m", "run_compiler", "model.json", "tuned.json", "--topk", str(args.topk), "--arch", args.arch]
    command3 = ["nnfusion", "model.onnx", "-f", "onnx", "-ftune_output_file=/dev/null", "-ftune_input_file=tuned.json"]
    if args.no_tc:
        command1.append("-ftc_rewrite=0")
        command3.append("-ftc_rewrite=0")
    if args.welder_base:
        command2.append("--nofusion")
    elif args.welder_none:
        command1.append("-fnofuse=1")
        command2.append("--nofusion")
        command3.append("-fnofuse=1")
    if args.skip_dot:
        command1.append('-ffusion_skiplist=Dot')
        command3.append('-ffusion_skiplist=Dot')

    subprocess.run(command1, check=True)
    subprocess.run(command2, check=True)
    subprocess.run(command3, check=True)
    subprocess.run(["rm", "-rf", "nnfusion_rt/cuda_codegen/build/"], check=True)
    subprocess.run(["cmake", "-S", "nnfusion_rt/cuda_codegen/", "-B", "nnfusion_rt/cuda_codegen/build/"], check=True)
    subprocess.run(["make", "-C", "nnfusion_rt/cuda_codegen/build/"], check=True)
