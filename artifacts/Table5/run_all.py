import subprocess
import os.path as osp


sub_dirs = ["resnet18", "resnet50", "vgg16", "unet"]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for model in sub_dirs:
        results.append(osp.join(prefix, model))
        model_strings.append(f"Model: {model}")
    return results, model_strings

if __name__ == "__main__":
    prefix = "../temp"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        ret = subprocess.run(["python", "../run_welder.py", sub_dir])
