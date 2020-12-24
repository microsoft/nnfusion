# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.insert(1, os.path.abspath("./src/python"))
os.environ["PATH"] = os.path.abspath(
    "./build/src/tools/nnfusion") + ":" + os.environ["PATH"]

from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import torch
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from nnf.runner import Runner
from nnf.trainer import Trainer


class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self._model = model

    def forward(self, input_ids, attention_mask, labels):
        out = self._model(input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        return out.loss


def test_runner():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', return_dict=True).to("cuda:0")
    wrapper = WrapperModel(model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch,
                         return_tensors='pt',
                         padding=True,
                         truncation=True)
    input_ids = encoding['input_ids'].to("cuda:0")
    attention_mask = encoding['attention_mask'].to("cuda:0")
    labels = torch.tensor([1, 0]).unsqueeze(0).to("cuda:0")
    loss = wrapper(input_ids, attention_mask, labels)

    nnf_flags = {"training_mode": 1, "kernel_fusion_level": 0, "blockfusion_level": 0}
    runner = Runner(wrapper, codegen_flags=nnf_flags)
    nnf_loss = runner(input_ids, attention_mask, labels)[0]
    print('pytorch_loss: ', loss)
    print('nnf_loss: ', nnf_loss)
    assert np.allclose(
        loss.cpu().detach().numpy(),
        nnf_loss.cpu().detach().numpy()), "Torch out: {}, NNF out: {}".format(
            loss, nnf_loss)


def for_test():
    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', return_dict=True).to(device)
    # model.load_state_dict(torch.load("/data/zimiao/models/bert_after_iter0.pt"))
    model.eval()
    # model.train()
    wrapper = WrapperModel(model)

    # for name, param in wrapper.named_parameters():
    #     print("{}: {}".format(name, param.requires_grad))
    #     assert param.requires_grad

    # sys.exit(0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch,
                         return_tensors='pt',
                         padding=True,
                         truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    labels = torch.tensor([1, 0]).unsqueeze(0).to(device)
    loss = wrapper(input_ids, attention_mask, labels)
    print(loss)

    # input_ids = torch.ones([2, 512], dtype=torch.int64).to(device)
    # attention_mask = torch.ones([2, 512], dtype=torch.int64).to(device)
    # labels = torch.ones([1, 2], dtype=torch.int64).to(device)

    codegen_flags = {
        # "codegen_debug": 1,
        # "kernel_fusion_level": 0,
        "blockfusion_level": 3
    }

    trainer = Trainer(wrapper, device=device, workdir="./tmp_bert_test_example", codegen_flags=codegen_flags)
    print("feeding")
    for i in range(100):
        pytorch_loss = trainer.run_by_pytorch(input_ids, attention_mask,
                                              labels)
        nnf_loss = trainer(input_ids, attention_mask, labels)
        print("iter ", i)
        print('pytorch_loss: ', pytorch_loss)
        print('nnf_loss: ', nnf_loss)
        # torch.save(model.state_dict(), "/data/zimiao/models/debug_nan/bert_after_iter{}.pt".format(i))


def export_onnx():
    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', return_dict=True).to(device)
    model.load_state_dict(
        torch.load("/data/zimiao/models/debug_nan/bert_after_iter45.pt"))
    model.eval()
    wrapper = WrapperModel(model)

    input_ids = torch.ones([1, 512], dtype=torch.int64).to(device)
    attention_mask = torch.ones([1, 512], dtype=torch.int64).to(device)
    labels = torch.ones([1, 1], dtype=torch.int64).to(device)

    print(wrapper(input_ids, attention_mask, labels))
    torch.onnx.export(wrapper.to(device), (input_ids, attention_mask, labels),
                      "/data/zimiao/models/bert_after_iter45.onnx",
                      opset_version=12,
                      input_names=["input0", "input1", "input2"],
                      output_names=["output_0"],
                      _retain_param_name=True,
                      do_constant_folding=False)


def get_grad():
    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', return_dict=True).to(device)
    model.load_state_dict(
        torch.load("/data/zimiao/models/bert_after_iter0.pt"))
    model.eval()
    wrapper = WrapperModel(model)

    input_ids = torch.ones([1, 512], dtype=torch.int64).to(device)
    attention_mask = torch.ones([1, 512], dtype=torch.int64).to(device)
    labels = torch.ones([1, 1], dtype=torch.int64).to(device)

    out = wrapper(input_ids, attention_mask, labels)
    out.backward()
    print(out)
    print(attention_mask.grad)


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_bert():
    train_texts, train_labels = read_imdb_split(
        '/data/zimiao/downloads/aclImdb/train')
    test_texts, test_labels = read_imdb_split(
        '/data/zimiao/downloads/aclImdb/test')

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          return_dict=True)
    model.to(device)
    # should switch to train()
    model.eval()
    wrapper = WrapperModel(model)
    # codegen_flags = {
    #     "kernel_fusion_level": 0
    # }

    codegen_flags = {}
    trainer = Trainer(wrapper,
                      device=device,
                      codegen_flags=codegen_flags,
                      workdir="./tmp_bert_train")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    sum_pt_loss = 0
    sum_nnf_loss = 0
    sum_iter = 0
    for epoch in range(100):
        for i, batch in enumerate(train_loader):
            if i % 100 == 0:
                if sum_iter == 0:
                    continue
                print("Epoch {}, batch {}, pt_loss {}, nnf_loss {}".format(
                    epoch, i, sum_pt_loss / sum_iter, sum_nnf_loss / sum_iter))
                sum_pt_loss = 0
                sum_nnf_loss = 0
                sum_iter = 0

            input_ids = batch['input_ids'].to(device)  # [8, 512]
            attention_mask = batch['attention_mask'].to(device)  # [8, 512]
            labels = batch['labels'].unsqueeze(0).to(device)  # [1, 8]
            if i == 0:
                print(input_ids.shape)
                print(attention_mask.shape)
                print(labels.shape)
            pt_loss = trainer.run_by_pytorch(input_ids, attention_mask, labels)
            # pt_loss = 0
            nnf_loss = trainer(input_ids, attention_mask, labels)
            sum_pt_loss += pt_loss
            sum_nnf_loss += nnf_loss
            sum_iter += 1
            # print("Epoch {}, batch {}, pt_loss {}, nnf_loss {}".format(epoch, i, sum_pt_loss/sum_iter, sum_nnf_loss/sum_iter))


if __name__ == "__main__":
    # test_runner()
    for_test()
    # train_bert()
    # export_onnx()