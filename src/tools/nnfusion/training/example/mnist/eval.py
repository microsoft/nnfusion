import torch, json, nnf, dtypes, dataprc


'''
Usage:
Codegen:
    ./build/src/tools/nnfusion/nnfusion models/onnx_autodiff/mnist_mlp.onnx -f onnx -p "batch:3" -ftraining_mode -fcodegen_debug=True -fextern_result_memory=True

Change file:
    1. CMakeLists.txt,19: cuda_add_library(${TARGET_NAME} SHARED ${SRC})
    2. nnfusion_rt.cu: removing first function call of cuda_init()

Build by cmake .. && make -j

Run evaluator by python nnf_py/eval.py

'''
cuda0 = torch.device('cuda:0')
json_path = 'para_info_eval.json'
loss_name = "loss"
out_name = "Softmax_27_0"

class NNFEvaluator:
    def __init__(self):
        self.parambyid = dict()
        # lastid = 0
        paramlist = list()
        siglist = list()
        torch.cuda.set_device(cuda0)
        self.loss_id = 0
        self.out_id = 0
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
                self.weight_ids.add(id)

            start = len(varinfo["input"]) + len(varinfo["weight"])
            for key in varinfo["output"]:
                id = int(varinfo["output"][key]["id"].split("outputs[")[1].split("]")[0]) + start
                shape = varinfo["output"][key]["shape"]
                dtype = varinfo["output"][key]["id"][2:].split("*")[0]
                self.parambyid[id] = (shape, dtype)
                if key == loss_name:
                    self.loss_id = id
                if key == out_name:
                    self.out_id = id
    
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
            # if key in self.weight_ids:
            #     torch.nn.init.uniform_(param)
            paramlist.append(param)

        self.paramlist = paramlist
        checkpoint = torch.load("torch_checkpoint")
        general_io_state = checkpoint["general_io_state"]
        for i in self.weight_ids:
            self.paramlist[i] = general_io_state[i]
        self.rt = nnf.Runtime()
        self.rt.init()

    def interation(self, inputs = dict()):
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
    def __init__(self):
        self.batch_size = 3
        self.data, self.test_data = dataprc.get_dataloader()
    def interation(self, batch, trainer):
        torch.set_printoptions(threshold=501408)
        # data, target
        batch = {"data": batch[0], "target": batch[1]}
        for k in batch:
            batch[k] = batch[k].to(cuda0)

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
                        raise Exception("Dtype is not suppported: %s" % (dtype))
                inputs[val] = torch.ones(shape, dtype=dtype, device=cuda0)

        trainer.interation(inputs = inputs)
        loss = trainer.paramlist[trainer.loss_id]
        out = trainer.paramlist[trainer.out_id]
        # print("out: ", out)
        pred = out.argmax(dim=1, keepdim=True)
        # print("pred: ", pred)
        # print("target: ", batch["target"])
        correct = pred.eq(batch["target"].view_as(pred)).sum().item()
        print("loss: ", loss, "correct: ", correct)
        return loss, correct


if __name__ == "__main__":
    trainer = NNFEvaluator()
    data = NaiveBertDataLoader()
    i = 1
    total_loss = 0
    total_correct = 0
    num = 0
    if len(data.test_data.dataset) % data.batch_size != 0:
        num = len(data.test_data.dataset) // data.batch_size
    for batch in data.test_data:
        print(i)
        i+= 1
        loss, correct = data.interation(batch, trainer)
        total_loss += loss
        total_correct += correct
        if i > num:
            break
    total_loss = total_loss/num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss, total_correct, data.batch_size * num,
        100. * total_correct / (data.batch_size * num)))
    # trainer.save()
    # trainer.finish()