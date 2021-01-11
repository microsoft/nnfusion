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

    nnf_flags = {
        "training_mode": 1,
        "kernel_fusion_level": 0,
        "blockfusion_level": 0
    }
    runner = Runner(wrapper, codegen_flags=nnf_flags)
    nnf_loss = runner(input_ids, attention_mask, labels)[0]
    print('pytorch_loss: ', loss)
    print('nnf_loss: ', nnf_loss)
    assert np.allclose(
        loss.cpu().detach().numpy(),
        nnf_loss.cpu().detach().numpy()), "Torch out: {}, NNF out: {}".format(
            loss, nnf_loss)


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


def process_data(imdb_data_dir):
    train_texts, train_labels = read_imdb_split(
        os.path.join(imdb_data_dir, "train"))
    test_texts, test_labels = read_imdb_split(
        os.path.join(imdb_data_dir, "test"))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


def pytorch_train_bert():
    # please download imdb data by yourself:
    # >wget http://ai.stanford.edu/\~amaas/data/sentiment/aclImdb_v1.tar.gz
    # >tar -xf aclImdb_v1.tar.gz
    imdb_data_dir = "/data/zimiao/downloads/aclImdb"
    train_dataset, _, _ = process_data(imdb_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(device)
    # should switch to train()
    model.eval()

    # optim = AdamW(model.parameters(), lr=5e-5)
    optim = torch.optim.SGD(model.parameters(), lr=0.0001)

    sum_loss = 0
    sum_iter = 0
    for epoch in range(100):
        for i, batch in enumerate(train_loader):
            if sum_iter == 100:
                print("Epoch {}, batch {}，loss {}".format(
                    epoch, i, sum_loss / sum_iter))
                sum_loss = 0
                sum_iter = 0

            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            sum_loss += loss
            sum_iter += 1


def train_bert():
    # please download imdb data by yourself:
    # >wget http://ai.stanford.edu/\~amaas/data/sentiment/aclImdb_v1.tar.gz
    # >tar -xf aclImdb_v1.tar.gz
    imdb_data_dir = "/path/to/aclImdb"
    train_dataset, _, _ = process_data(imdb_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    device = "cuda:0"
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          return_dict=True)
    model.to(device)
    #TODO: should switch to train(), but nnf dropout kernel is not ready
    model.eval()
    wrapper = WrapperModel(model)

    codegen_flags = {"kernel_fusion_level": 0, "blockfusion_level": 0}
    trainer = Trainer(wrapper, device=device, codegen_flags=codegen_flags)

    sum_nnf_loss = 0
    sum_iter = 0
    for epoch in range(100):
        for i, batch in enumerate(train_loader):
            if sum_iter == 100:
                print("Epoch {}, batch {}, nnf_loss {}".format(
                    epoch, i, sum_nnf_loss / sum_iter))
                sum_nnf_loss = 0
                sum_iter = 0

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(0).to(device)

            nnf_loss = trainer(input_ids, attention_mask, labels)

            sum_nnf_loss += nnf_loss
            sum_iter += 1


if __name__ == "__main__":
    # test_runner()
    train_bert()