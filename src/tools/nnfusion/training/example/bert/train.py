import torch, json, nnf, dtypes, dataprc


'''
Usage:
Codegen:
    ./build/src/tools/nnfusion/nnfusion models/onnx_autodiff/bert_grad_with_cost_addition_input.onnx -f onnx -p "batch:3;sequence:512;dynamic_prediction_count:20" -fautodiff -ftraining_mode -fcodegen_debug=True -fextern_result_memory=True

Change file:
    1. CMakeLists.txt,19: cuda_add_library(${TARGET_NAME} SHARED ${SRC})
    2. nnfusion_rt.cu: removing first function call of cuda_init()

Build by cmake .. && make -j

Run trainer by python nnf_py/train.py

'''
cuda0 = torch.device('cuda:0')
json_path = 'para_info.json'
loss_name = "total_loss"

class NNFTrainer:
    def __init__(self):
        self.parambyid = dict()
        # lastid = 0
        paramlist = list()
        siglist = list()
        torch.cuda.set_device(cuda0)
        self.loss_id = 0
        self.inputs_name_id = dict()
        self.weight_ids = set()

        # Should load info from var_info.json
        with open(json_path) as json_file:
            varinfo = json.load(json_file)
            for key in varinfo["input"]:
                id = int(varinfo["input"][key]["id"].split("inputs[")[1].split("]")[0])
                shape = varinfo["input"][key]["shape"]
                dtype = varinfo["input"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)
                self.inputs_name_id[key] = id

            for key in varinfo["weight"]:
                id = int(varinfo["weight"][key]["id"].split("inputs[")[1].split("]")[0])
                shape = varinfo["weight"][key]["shape"]
                dtype = varinfo["weight"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)

            start = len(varinfo["input"]) + len(varinfo["weight"])
            for key in varinfo["output"]:
                id = int(varinfo["output"][key]["id"].split("outputs[")[1].split("]")[0]) + start
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
            param = torch.ones(shape, dtype=dtype, device=cuda0)
            if key in self.weight_ids:
                torch.nn.init.uniform_(param)
            paramlist.append(param)

        self.paramlist = paramlist
        self.rt = nnf.Runtime()
        self.rt.init()

    def interation(self, inputs = dict()):
        # replacing param with data tensor
        for i in inputs:
            self.paramlist[i] = inputs[i]
        self.rt.feed(tensors=self.paramlist)

    def finish(self):
        self.rt.free()

    def save(self):
        torch.save({"general_io_state": self.paramlist}, "torch_checkpoint")

class NaiveBertDataLoader:
    def __init__(self):
        self.batch_size = 3
        self.data, _, _= dataprc.get_dataloader()

    def interation(self, batch, trainer):
        # 78~85 are inputs
        # input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights, [next_sentence_labels], input4
        for k in batch:
            batch[k] = batch[k].to(cuda0)

        # labels = batch["label"] #next_sentence_labels
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"] # input_mask
        # token_type_ids = batch["token_type_ids"] # segment_ids
        # masked_lm_positions = batch["masked_lm_positions"]
        # masked_lm_ids = batch["masked_lm_ids"]
        # masked_lm_weight = batch[ "masked_lm_weights"]

        name_map = {"input1": "input_ids", "input2": "token_type_ids", "input3": "attention_mask",
        "masked_lm_positions": "masked_lm_positions", "masked_lm_ids": "masked_lm_ids",
        "masked_lm_weights": "masked_lm_weights", "next_sentence_labels": "label"}

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
                        raise Exception("Dtype is not suppported: %s" % (dtype))
                inputs[val] = torch.ones(shape, dtype=dtype, device=cuda0)

        # # input1, input_ids, 1536
        # inputs[78] = input_ids
        # # input2, segment_ids, 1536
        # inputs[79] = token_type_ids
        # # input3, input_mask, 1536
        # inputs[80] = attention_mask
        # # masked_lm_positions, 60
        # inputs[81] = masked_lm_positions
        # # masked_lm_ids, 60
        # inputs[82] = masked_lm_ids
        # # masked_lm_weights, 60
        # inputs[83] = masked_lm_weight
        # # [next_sentence_labels], 3
        # inputs[84] = labels
        # # input4, dummy node, 1572864
        # inputs[85] = torch.ones(1572864, dtype=torch.float, device=cuda0)

        trainer.interation(inputs = inputs)
        loss = trainer.paramlist[trainer.loss_id]

        print("total loss: ", loss)

        return loss


if __name__ == "__main__":
    trainer = NNFTrainer()
    data = NaiveBertDataLoader()
    i = 0
    for batch in data.data:
        print(i)
        i+= 1
        data.interation(batch, trainer)

    trainer.save()
    trainer.finish()
