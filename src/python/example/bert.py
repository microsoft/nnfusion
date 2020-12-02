# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.insert(1, os.path.abspath("./src/python"))
os.environ["PATH"] = os.path.abspath(
    "./build/src/tools/nnfusion") + ":" + os.environ["PATH"]

import numpy as np
import torch
torch.manual_seed(0)
from torch import nn
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
    # model.train()
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

    nnf_flags = {"training_mode": 1}
    runner = Runner(wrapper, codegen_flags=nnf_flags)
    nnf_loss = runner(input_ids, attention_mask, labels)[0]
    assert np.allclose(
        loss.cpu().detach().numpy(),
        nnf_loss.cpu().detach().numpy()), "Torch out: {}, NNF out: {}".format(
            loss, nnf_loss)


def train_bert():
    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', return_dict=True).to(device)
    # model.train()
    wrapper = WrapperModel(model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch,
                         return_tensors='pt',
                         padding=True,
                         truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    labels = torch.tensor([1, 0]).unsqueeze(0).to(device)
    # loss = wrapper(input_ids, attention_mask, labels)
    # print(loss)

    trainer = Trainer(wrapper, device=device)
    print("feeding")
    for i in range(10):
        pytorch_loss = trainer.run_by_pytorch(input_ids, attention_mask,
                                              labels)
        nnf_loss = trainer(input_ids, attention_mask, labels)
        print("iter ", i)
        print('pytorch_loss: ', pytorch_loss)
        print('nnf_loss: ', nnf_loss)

    torch.save(model.state_dict(), "/tmp/bert.pt")


if __name__ == "__main__":
    test_runner()
    train_bert()