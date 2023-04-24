import os.path as osp
import json
from welder.engine import load_model
import numpy as np
import subprocess

sub_dirs = [
    ["NAFNet/64"],
    ["NAFNet_welder_base"],
    ["NAFNet_welder_none"],
    ["vit/64"],
    ["vit_welder_base"],
    ["vit_welder_none"],
    ["swin/64"],
    ["swin_welder_base"],
    ["swin_welder_none"],
]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for vals in sub_dirs:
        model_strings.append(f"Model: {vals[0]}")
        results.append(osp.join(prefix, *map(str, vals)))
    return results, model_strings

if __name__ == "__main__":
    prefix = "../temp"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        all_od = set()
        with open(osp.join(sub_dir, "tuned.json")) as f:
            obj = json.loads(f.read())
            kern_count = len(obj)
            for op in obj:
                if "output_desc" in op:
                    for od in op["output_desc"]:
                        all_od.add(tuple(od))
                else:
                    all_od.add((op['nodes'][0], 0))

        ordered_nodes = load_model(osp.join(sub_dir, "model.json"))

        total_size = 0
        for node in ordered_nodes:
            if node.get_tag("memcpy"):
                kern_count -= 1
                continue
            if node.is_output():
                continue
            node_id = int(node.name[node.name.rfind("_")+1:])
            for i in range(node.num_outputs()):
                if (node_id, i) in all_od:
                    total_size += np.prod(node.get_shape(i)) * (node.get_dtype(i).bits // 8)
        print("IRS: ", total_size / (1024 * 1024), "MB")
        print("Kernel Count: ", kern_count)
        subprocess.run(["python", "../run_welder.py", sub_dir], check=True)
