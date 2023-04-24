import subprocess
import os.path as osp

data_prefix = ""

sub_dirs = [
    ["notc_mobilenet", (1, 64)],
    ["notc_NAFNet", (1, 64)],
    ["notc_vit", (1, 64)],
    ["notc_bert", (1, 64)],
    ["notc_mobilevit", (1, 64)],
    ["notc_swin", (1, 64)],
    ["notc_Conformer", (1, 64)],
    ["notc_BSRN", (1, )],
    ["notc_restormer", (1, )],
    ["notc_NeRF", (1, )]
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
    prefix = "../temp/"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        ret = subprocess.run(["python", "../run_welder.py", sub_dir])
