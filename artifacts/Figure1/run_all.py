import subprocess
import os.path as osp

sub_dirs = [
    ["bert", (64, ), ("fp32", )],
    ["vit", (64, ), ("fp32", )],
    ["swin", (64, ), ("fp32", )],
    ["mobilevit", (64, ), ("fp32", )],
    ["NAFNet", (64, ), ("fp32", )],
    ["NeRF", (1, ), ("fp32", )]
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

# sudo required
if __name__ == "__main__":
    prefix = "/sharepoint/e2e/"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        base_comamnd = "python profile_onnxrt.py --prefix " + sub_dir
        command1 = "nvprof --profile-from-start off --log-file profile --csv " + base_comamnd
        subprocess.run(command1.split(), check=True)
        command2 = "nvprof --profile-from-start off --log-file metrics --csv --metrics gld_throughput,gst_throughput,flop_count_sp " + base_comamnd
        subprocess.run(command2.split(), check=True)
        command3 = "python process_metrics.py"
        subprocess.run(command3.split(), check=True)
