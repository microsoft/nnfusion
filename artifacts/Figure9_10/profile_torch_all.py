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
                results.append((model, b, t))
                model_strings.append(f"Model: {model} BS: {b}, dtype: {t}")
    return results, model_strings


if __name__ == "__main__":
    prefix = ""
    for task, model_string in zip(*get_sub_dirs(prefix)):
        model, batch_size, dtype = task
        print("Running", model_string)
        args = ["python", "../run_torch.py", model, "--bs", str(batch_size)]
        if dtype == "fp16":
            args.append("--fp16")
        ret = subprocess.run(args)
