import subprocess
import os.path as osp

sub_dirs = [
    ["mobilenet", (1, 64), ("fp32", "fp16")],
    ["NAFNet", (1, 64), ("fp32", "fp16")],
    ["vit", (1, 64), ("fp32", "fp16")],
    ["bert", (1, 64), ("fp32", "fp16")],
    ["mobilevit", (1, 64), ("fp32", "fp16")],
    ["swin", (1, 64), ("fp32", "fp16")],
    ["Conformer", (1, 64), ("fp32", "fp16")],
    ["BSRN", (1, ), ("fp32", "fp16")],
    ["restormer", (1, ), ("fp32", "fp16")],
    ["NeRF", (1, ), ("fp32", "fp16")]
]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for model, bs, tp in sub_dirs:
        osp.join(prefix, model)
        for b in bs:
            for t in tp:
                suffix = str(b) + ("_fp16" if t == "fp16" else "")
                results.append(osp.join(prefix, model, suffix))
                model_strings.append(f"Model: {model} BS: {b}, dtype: {t}")
    return results, model_strings


if __name__ == "__main__":
    prefix = "/sharepoint/e2e/"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        ret = subprocess.run(["python", "../run_trt.py", "--prefix", sub_dir])
