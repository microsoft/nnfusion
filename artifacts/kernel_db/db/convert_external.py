# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script helps you to insert specific kernels (specs in json files) into our kernel_db
Some more explains to customize it for your needs
    param_list : specify the parameters and their data types for a certain type of kernel
    gen_key    : generate identifier of kernel specification, including I/O shapes, types and more
    gen_config : convert kernel specification to the db format 
    insert_db  : insert the parsed kernel into kernel db
"""

import json
import sys
import sqlite3
import os
import math
import argparse

from .cuparse import parse as code_parse
from .profile import prepare_file, log_sync, profile, prod

# db_path = os.environ['HOME'] + "/rocm_db"
db_path = os.environ['HOME'] + "/.cache/nnfusion/"
db_name = "kernel_cache.db"

def gen_config(op_type, identifier, kernel, shared_memory, num_sync):
    # the entries to retrive parameters depend on spec of json files
    config = {
        "op_type": op_type,
        "identifier": identifier,
        "function_body": "",
        "shared_memory": shared_memory,
        "num_sync": num_sync,
        "blockDim": kernel["blockDim"],
        "gridDim": kernel["gridDim"],
    }
    return config


def insert_db(name, resource, platform="CUDA_GPU", tags="", profile="Tesla V100-PCIE-16GB:1"):
    # Todo: More tags could be used to store multiple implementations with the same kernel specs
    in_file = open(name + ".cu")
    json_file = open(name + ".json")

    data = json.load(json_file)
    block_function_body = in_file.read()
    data["block_function_body"] = block_function_body

    
    key = data["function_body"]
    identifier = gen_key(data)
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

    conn = sqlite3.connect(db_path + db_name)
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

    identifier = gen_key(data)
    print(identifier)
    # overwrite the same implementation
    c.execute("DELETE FROM KernelCache WHERE Key = ?", (key,))
    c.execute("INSERT INTO KernelCache (Key,Identifier,OpType,Attributes,Source,DeviceType,Function,Tags,Miscs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (key, identifier, op_type, attributes, source, device_type, function, tags, miscs))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for converting json kernel file to kernel_cache.db')
    parser.add_argument('--json_file', type=str, required=True,
                        help='roller json kernel file path')
    parser.add_argument('--db_path', type=str, required=True,
                        help='the path to write kernel_cache.db')
    args = parser.parse_args()
    db_path = args.db_path
    if not os.path.isdir(db_path):
        os.mkdir(db_path)
    json_file = open(args.json_file)
    kernels = json.load(json_file)
    # input json file could contain one or more kernels
    if "op_type" in kernels:
        kernels = [kernels]
    for kernel in kernels:
        op_type = kernel["op_type"]

        # parse and clean up the cuda code to get some specific information
        func_body, shared_memory, new_code, sync_code, signature = code_parse(
            kernel["code"], param_list[op_type])

        config = gen_config(op_type, kernel, shared_memory, num_sync=0)

        prepare_file(signature, sync_code, config,
                     db_path + "profile/", parse=True)
        num_sync = log_sync(signature, db_path + "profile/")
        config["num_syncthreads"] = num_sync
        config["function_body"] = func_body

        # feel free to customize the repo name you want
        name = kernel["tvm_func_name"].replace("_kernel0", "")
        operator_path = db_path + op_type + "_db/"
        if not os.path.isdir(operator_path):
            os.mkdir(operator_path)
        with open(operator_path + name + ".json", "w+") as f:
            json.dump(config, f)
        with open(operator_path + name + ".cu", "w+") as f:
            f.write(new_code)

        default_tags = ""
        default_tags += "KernelEmitter,CudaEmitter,BlockCudaEmitter"
        if (op_type == "Dot"):
            # Todo: move the transpose information into identifier
            default_tags += kernel["parameters"]["transpose_A"] * \
                ",transA" + kernel["parameters"]["transpose_B"]*",transB"

        # apply rules that every 32 threads will be formed as a warp
        resource = math.ceil(
            prod(config["blockDim"])/32)*32 * prod(config["gridDim"])

        prepare_file(signature, kernel["code"], config, db_path + "profile/")
        profile_info = profile(signature, db_path + "profile/")
        print(profile_info, resource, config["num_syncthreads"])
        insert_db(operator_path + name, resource,
                  tags=default_tags, profile=profile_info)
