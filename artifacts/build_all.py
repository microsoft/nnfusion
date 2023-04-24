import subprocess
import os.path as osp
import os

sub_dirs = [
    "mobilenet/1",
    "mobilenet/64",
    "mobilenet/1_fp16",
    "mobilenet/64_fp16",
    "NAFNet/1",
    "NAFNet/64",
    "NAFNet/1_fp16",
    "NAFNet/64_fp16",
    "vit/1",
    "vit/64",
    "vit/1_fp16",
    "vit/64_fp16",
    "bert/1",
    "bert/64",
    "bert/1_fp16",
    "bert/64_fp16",
    "mobilevit/1",
    "mobilevit/64",
    "mobilevit/1_fp16",
    "mobilevit/64_fp16",
    "swin/1",
    "swin/64",
    "swin/1_fp16",
    "swin/64_fp16",
    "Conformer/1",
    "Conformer/64",
    "Conformer/1_fp16",
    "Conformer/64_fp16",
    "BSRN/1",
    "BSRN/1_fp16",
    "restormer/1",
    "restormer/1_fp16",
    "NeRF/1",
    "NeRF/1_fp16",

    "NAFNet_welder_base",
    "swin_welder_base",
    "vit_welder_base",
    "NAFNet_welder_none",
    "vit_welder_none",
    "swin_welder_none",

    "resnet18",
    "resnet50",
    "vgg16",
    "unet",
]

sub_dirs2 = [
    "notc_mobilenet/1",
    "notc_mobilenet/64",
    "notc_NAFNet/1",
    "notc_NAFNet/64",
    "notc_vit/1",
    "notc_vit/64",
    "notc_bert/1",
    "notc_bert/64",
    "notc_mobilevit/1",
    "notc_mobilevit/64",
    "notc_Conformer/1",
    "notc_Conformer/64",
    "notc_swin/1",
    "notc_swin/64",
    "notc_BSRN/1",
    "notc_restormer/1",
    "notc_NeRF/1",
]

if __name__ == "__main__":
    prefix = "./temp/"
    cur_dir = os.path.abspath(".")
    for sub_dir in sub_dirs:
        print("Building", sub_dir)
        os.chdir(osp.join(prefix, sub_dir))
        command = ["nnfusion", "model.onnx", "-f", "onnx", "-ftune_output_file=/dev/null", "-ftune_input_file=tuned.json"]
        if sub_dir in ["bert/1", "bert/64", "swin/1_fp16"]:
            command.append('-ffusion_skiplist=Dot')
        if "welder_none" in sub_dir:
            command.append('-fnofuse=1')
        subprocess.run(command, check=True, capture_output=True)
        subprocess.run(["rm", "-rf", "nnfusion_rt/cuda_codegen/build/"], check=True)
        subprocess.run(["cmake", "-S", "nnfusion_rt/cuda_codegen/", "-B", "nnfusion_rt/cuda_codegen/build/"], check=True, capture_output=True)
        subprocess.run(["make", "-C", "nnfusion_rt/cuda_codegen/build/"], check=True, capture_output=True)
        os.chdir(cur_dir)

    for sub_dir in sub_dirs2:
        print("Building", sub_dir)
        orig_model = osp.abspath(osp.join(prefix, sub_dir[5:] + "_fp16", "model.onnx"))
        os.chdir(osp.join(prefix, sub_dir))
        command = ["nnfusion", orig_model, "-f", "onnx", "-ftune_output_file=/dev/null", "-ftune_input_file=tuned.json", "-ftc_rewrite=0"]
        subprocess.run(command, check=True, capture_output=True)
        subprocess.run(["rm", "-rf", "nnfusion_rt/cuda_codegen/build/"], check=True)
        subprocess.run(["cmake", "-S", "nnfusion_rt/cuda_codegen/", "-B", "nnfusion_rt/cuda_codegen/build/"], check=True, capture_output=True)
        subprocess.run(["make", "-C", "nnfusion_rt/cuda_codegen/build/"], check=True, capture_output=True)
        os.chdir(cur_dir)
