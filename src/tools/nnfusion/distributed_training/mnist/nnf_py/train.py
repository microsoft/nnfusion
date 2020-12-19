# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import json
import nnf
import dataprc
import math
import sys


json_path = './para_info.json'
loss_name = "loss"


class NNFTrainer:
    def __init__(self, plan_file_path):

        self.rt = nnf.Runtime()
        self.rt.init(plan_file_path)
        self.device_id = self.rt.device_id()
        self.world_size = self.rt.world_size()
        self.cuda_device_id = 'cuda:' + str(self.rt.device_id())
        print("i am running @ local device: ", self.cuda_device_id)
        self.cuda_device = torch.device(self.cuda_device_id)
        torch.cuda.set_device(self.cuda_device)

        self.parambyid = dict()
        # lastid = 0
        paramlist = list()
        # siglist = list()
        self.loss_id = 0
        self.inputs_name_id = dict()
        # self.weight_ids = set()
        self.weight_name_id = dict()  # {name: [w_id, b_id]}

        # Should load info from var_info.json
        with open(json_path) as json_file:
            varinfo = json.load(json_file)
            for key in varinfo["input"]:
                id = int(varinfo["input"][key]["id"].split(
                    "inputs[")[1].split("]")[0])
                shape = varinfo["input"][key]["shape"]
                dtype = varinfo["input"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)
                self.inputs_name_id[key] = id

            for key in varinfo["weight"]:
                id = int(varinfo["weight"][key]["id"].split(
                    "inputs[")[1].split("]")[0])
                shape = varinfo["weight"][key]["shape"]
                dtype = varinfo["weight"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)
                # self.weight_ids.add(id)
                if key.endswith(".weight"):
                    name = key[: key.rindex(".weight")]
                    if name not in self.weight_name_id:
                        self.weight_name_id[name] = [id, None]
                    else:
                        self.weight_name_id[name][0] = id
                elif key.endswith(".bias"):
                    name = key[: key.rindex(".bias")]
                    if name not in self.weight_name_id:
                        self.weight_name_id[name] = [None, id]
                    else:
                        self.weight_name_id[name][1] = id

            start = len(varinfo["input"]) + len(varinfo["weight"])
            for key in varinfo["output"]:
                id = int(varinfo["output"][key]["id"].split(
                    "outputs[")[1].split("]")[0]) + start
                shape = varinfo["output"][key]["shape"]
                dtype = varinfo["output"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)
                if key == loss_name:
                    self.loss_id = id

        for key in sorted(self.parambyid):
            # inititalizer
            (shape, dtype) = self.parambyid[key]
            if dtype == "float":
                dtype = torch.float
            else:
                if dtype == "int64_t":
                    dtype = torch.int64
                else:
                    raise Exception("Dtype is not suppported: %s" % (dtype))
            param = torch.ones(shape, dtype=dtype, device=self.cuda_device)
            # if key in self.weight_ids:
            #     torch.nn.init.uniform_(param)
            paramlist.append(param)

        self.paramlist = paramlist

        for v in self.weight_name_id.values():
            w_id, b_id = v
            torch.nn.init.kaiming_uniform_(
                self.paramlist[w_id], a=math.sqrt(5))
            if b_id is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.paramlist[w_id])
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.paramlist[b_id], -bound, bound)

    def interation(self, inputs=dict()):
        # replacing param with data tensor
        # torch.set_printoptions(threshold=501408)
        for i in inputs:
            self.paramlist[i] = inputs[i]
        self.rt.feed(tensors=self.paramlist)

    def finish(self):
        self.rt.free()

    def save(self):
        torch.save({"general_io_state": self.paramlist}, "torch_checkpoint")


class NaiveBertDataLoader:
    def __init__(self, device_id, world_size):
        self.batch_size = 3
        self.data, _ = dataprc.get_dataloader(device_id, world_size)

    def interation(self, batch, trainer):
        # data, target
        batch = {"data": batch[0], "target": batch[1]}
        for k in batch:
            batch[k] = batch[k].to(trainer.cuda_device)

        name_map = {"data": "data", "target": "target"}

        inputs = dict()
        for key, val in trainer.inputs_name_id.items():
            if key in name_map and name_map[key] in batch:
                inputs[val] = batch[name_map[key]]
            else:
                (shape, dtype) = trainer.parambyid[val]
                if dtype == "float":
                    dtype = torch.float
                else:
                    if dtype == "int64_t":
                        dtype = torch.int64
                    else:
                        raise Exception(
                            "Dtype is not suppported: %s" % (dtype))
                inputs[val] = torch.ones(
                    shape, dtype=dtype, device=trainer.cuda_device)

        trainer.interation(inputs=inputs)
        loss = trainer.paramlist[trainer.loss_id]
        print("total loss: ", loss)

        return loss


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception('no plan is given')
    trainer = NNFTrainer(sys.argv[1])
    data = NaiveBertDataLoader(trainer.device_id, trainer.world_size)
    i = 1
    for batch in data.data:
        print(i)
        i += 1
        data.interation(batch, trainer)
    trainer.save()
    trainer.finish()
