# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

from transformers import BertForSequenceClassification
from transformers import BertTokenizer

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('/data/zimiao/downloads/aclImdb/train')
test_texts, test_labels = read_imdb_split('/data/zimiao/downloads/aclImdb/test')

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased')
model.to(device)
# should switch to train()
model.eval()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)
optim = torch.optim.SGD(model.parameters(), lr=0.0001)


sum_loss = 0
sum_iter = 0
for epoch in range(3):
    for i, batch in enumerate(train_loader):
        if i % 100 == 0:
            if sum_iter == 0:
                continue
            print("Epoch {}, batch {}，loss {}".format(epoch, i, sum_loss/sum_iter))
            sum_loss = 0
            sum_iter = 0

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device) # [8, 512]
        attention_mask = batch['attention_mask'].to(device) # [8, 512]
        labels = batch['labels'].to(device) # [8]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        sum_loss += loss
        sum_iter += 1        
        loss.backward()
        optim.step()

model.eval()