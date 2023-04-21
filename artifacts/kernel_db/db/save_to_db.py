import json
import sys
import sqlite3
import os
import math
import argparse
import re
import getpass

from .cuparse import parse as code_parse

db_path = os.environ['HOME'] + "/.cache/nnfusion"
db_name = "kernel_cache.db"

def parse_from_identifier(identifier):
    op_type = identifier.split("[")[0]
    if op_type.startswith("Matched_Pattern"):
        inner_ops = identifier[identifier.find("[") + 1: -1]
        inner_ops = re.findall(r"(\w+\[.*?\])", inner_ops)
        inner_ops = [parse_from_identifier(op) for op in inner_ops]
        if (op_type == "Matched_Pattern(Convolution-Add-Relu)"):
            # the second "1" is for parsing tuple (op_type, in_shape, out_shape)
            in_shape = [inner_ops[0][1][0], inner_ops[0][1][1], inner_ops[1][1][1]]
        else:
            raise NotImplementedError
        out_shape = inner_ops[-1][2]
        return op_type, in_shape, out_shape
    else:
        shape_info = re.search("([0-9,;]+)", identifier).group(1)
        shapes = shape_info.split(";")
        for i in range(len(shapes)):
            shape = shapes[i].split(",")
            shapes[i] = [int(x) for x in shape]
        # Assume only one output
        return op_type, shapes[:-1], [shapes[-1]]


def clean_up(source):
    match = re.search("__launch_bounds__\([0-9]*\) ", source)
    if match:
        lb = match.group()
        source = source.replace(lb, "")
    source = source[source.find("extern"):]
    return source


def insert_db(name, resource, identifier, platform="CUDA_GPU", tags="", profile="Tesla V100-PCIE-16GB:1"):
    print("start insert db", flush=True)
    # Todo: More tags could be used to store multiple implementations with the same kernel specs
    in_file = open(name + ".cu")
    json_file = open(name + ".json")

    data = json.load(json_file)
    block_function_body = in_file.read()
    data["block_function_body"] = block_function_body

    
    key = identifier + "\n" + data["function_body"]
    op_type = data["op_type"]
    source = "External"
    device_type = platform

    attributes_dict = {}
    attributes_dict.update({"input_shape": data["in_shape"]})
    attributes_dict.update({"output_shape": data["out_shape"]})
    if data.get("parameters") != None:
        attributes_dict.update({"parameters": data["parameters"]})
    attributes = json.dumps(attributes_dict)

    function_dict = {}
    function_dict.update({"function_signature": data["function_signature"]})
    function_dict.update({"function_body": data["function_body"]})
    function_dict.update({"grid_dim": data["gridDim"]})
    function_dict.update({"block_dim": data["blockDim"]})
    function_dict.update({"block_function_body": data["block_function_body"]})
    function_dict.update({"shared_memory": data["shared_memory"]})
    function_dict.update({"num_syncthreads": data["num_syncthreads"]})
    function = json.dumps(function_dict)

    miscs_dict = {}
    profile_dict = {"time": profile, "resource": resource}
    miscs_dict.update({"external_profile": profile_dict})
    miscs = json.dumps(miscs_dict)

    conn = sqlite3.connect(os.path.join(db_path, db_name))
    c = conn.cursor()

    create_sql = "create table if not exists KernelCache (\
            Key        TEXT NOT NULL,\
            Identifier TEXT NOT NULL,\
            OpType     TEXT NOT NULL,\
            Attributes TEXT DEFAULT '',\
            Source     TEXT DEFAULT 'External',\
            DeviceType TEXT NOT NULL,\
            Function   TEXT NOT NULL,\
            Tags       TEXT DEFAULT '',\
            Miscs      TEXT DEFAULT '',\
            PRIMARY KEY(Key)\
            )"

    c.execute(create_sql)

    # overwrite the same implementation
    c.execute("DELETE FROM KernelCache WHERE Identifier = ?", (identifier,))
    c.execute("INSERT INTO KernelCache (Key,Identifier,OpType,Attributes,Source,DeviceType,Function,Tags,Miscs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (key, identifier, op_type, attributes, source, device_type, function, tags, miscs))
    conn.commit()
    conn.close()
    print("end insert db", flush=True)
    os.system(f"rsync -avz ~/.cache/nnfusion/* /tmp/{getpass.getuser()}/")


# device_type: CUDA_GPU, ROCM_GPU
def save_to_db(identifier, source, grid_size, block_size, device_type="CUDA_GPU"):
    if not os.path.exists(db_path):
        os.mkdir(db_path)

    if device_type == "CUDA_GPU":
        from .profile_cuda import prepare_file, log_sync, profile, prod
    elif device_type == "ROCM_GPU":
        from .profile_rocm import prepare_file, log_sync, profile, prod
    else:
        raise NotImplementedError

    # parse and clean up the cuda code to get some specific information
    source = clean_up(source)
    func_source, func_sig, func_body, shared_memory, new_code, sync_code, signature = code_parse(source)

    op_type, in_shape, out_shape = parse_from_identifier(identifier)
    config = {
        "op_type": op_type,
        "function_body": "",
        "shared_memory": shared_memory,
        "num_sync": 0,
        "blockDim": block_size,
        "gridDim": grid_size,
        "in_shape": in_shape,
        "out_shape": out_shape,
        "function_signature": func_sig.replace("default_function_kernel0", "")
    }

    prepare_file(signature, sync_code, config,
                    os.path.join(db_path, "profile"), parse=True)
    num_sync = log_sync(signature, os.path.join(db_path, "profile"))
    config["num_syncthreads"] = num_sync
    config["function_body"] = func_body

    # feel free to customize the repo name you want
    name = identifier
    operator_path = os.path.join(db_path, op_type + "_db")
    if not os.path.isdir(operator_path):
        os.mkdir(operator_path)
    if len(name) > 240: name = name[:240]
    with open(operator_path + "/" + name + ".json", "w+") as f:
        json.dump(config, f)
    with open(operator_path + "/" + name + ".cu", "w+") as f:
        f.write(new_code)

    default_tags = ""
    default_tags += "KernelEmitter,CudaEmitter,BlockCudaEmitter"

    # apply rules that every 32 threads will be formed as a warp
    resource = math.ceil(
        prod(config["blockDim"])/32)*32 * prod(config["gridDim"])

    prepare_file(signature, func_source, config, os.path.join(db_path, "profile"))
    profile_info = profile(signature, os.path.join(db_path, "profile"))
    print(profile_info, resource, config["num_syncthreads"])
    insert_db(operator_path + "/" + name, resource, identifier,
              platform=device_type, tags=default_tags, profile=profile_info)
    # repo.index.add(db_path)
    # repo.index.commit(f"{identifier} with {profile_info}")
    return True


if __name__ == '__main__':
    print(parse_from_identifier("Matched_Pattern(Convolution-Add-Relu)[Convolution[1,1024,14,14;256,1024,1,1;1,256,14,14floatfloatfloatStrides{1, 1}Strides{1, 1}CoordinateDiff{0, 0}]Add[1,256,14,14;1,256,14,14;1,256,14,14floatfloatfloat]Relu[1,256,14,14;1,256,14,14floatfloat]]"))