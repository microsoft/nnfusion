import subprocess
import os.path as osp

data_prefix = ""

sub_dirs = [
    ["mobilenet", (1, 64)],
    ["NAFNet", (1, 64)],
    ["vit", (1, 64)],
    ["bert", (1, 64)],
    ["mobilevit", (1, 64)],
    ["swin", (1, 64)],
    ["Conformer", (1, 64)],
    ["BSRN", (1, )],
    ["restormer", (1, )],
    ["NeRF", (1, )]
]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for model, bs in sub_dirs:
        osp.join(prefix, model)
        for b in bs:
            suffix = str(b)
            results.append(osp.join(prefix, model, suffix))
            model_strings.append(f"Model: {model} BS: {b}")
    return results, model_strings

if __name__ == "__main__":
    prefix = "/sharepoint/Figure10b/"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        ret = subprocess.run(["python", "../run_welder.py", sub_dir])
