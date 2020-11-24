import sys
import os
sys.path.insert(1, os.path.abspath("./src/python"))
os.environ["PATH"] = os.path.abspath(
    "./build/src/tools/nnfusion") + ":" + os.environ["PATH"]

import numpy as np
import torch
from torch import nn
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from nnf.runner import Runner


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

    external_weights = {
        "_model.bert.embeddings.position_ids":
        torch.arange(512).expand((1, -1)).to("cuda:0")
    }
    nnf_flags = {"training_mode": 1}
    runner = Runner(wrapper, external_weights=external_weights, codegen_flags=nnf_flags)
    nnf_loss = runner(input_ids, attention_mask, labels)[0]
    assert np.allclose(
        loss.cpu().detach().numpy(),
        nnf_loss.cpu().detach().numpy()), "Torch out: {}, NNF out: {}".format(
            loss, nnf_loss)


if __name__ == "__main__":
    test_runner()
# from transformers import AdamW
# optimizer = AdamW(model.parameters(), lr=1e-5)

# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# text_batch = ["I love Pixar.", "I don't care for Pixar."]
# encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']
# print(attention_mask)
# print(input_ids)

# labels = torch.tensor([1,0]).unsqueeze(0)
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# print(outputs)
# loss = outputs.loss
# loss.backward()
# optimizer.step()