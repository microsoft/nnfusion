import subprocess
import os.path as osp
import time

sub_dirs = [
    ["mobilenet", (1, ), ("fp32", )],
    ["bert", (1, ), ("fp32", )],
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
    prefix = "../temp"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        start_time = time.time()
        ret = subprocess.run(["python", "-m", "run_compiler", osp.join(sub_dir, "model.json"), "/dev/null", "--topk", "20", "--arch", "V100"],
                            capture_output=True)
        end_time = time.time()
        print(f"Time used: {end_time-start_time}s")
