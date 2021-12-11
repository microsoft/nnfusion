from typing import Dict
from functools import partial

import lineflow as lf
import lineflow.datasets as lfds
import lineflow.cross_validation as lfcv

import torch
from torch.utils.data import *

from transformers import BertTokenizer

import collections
import random

MAX_LEN = 512
rng = random.Random(12345)
max_predictions_per_seq = 20
masked_lm_prob = 0.15

def create_masked_lm_predictions(input_ids, masked_lm_prob,
                                 max_predictions_per_seq, tokenizer: BertTokenizer, rng):
    cand_indexes = []
    for (i, input_id) in enumerate(input_ids):
        token = tokenizer.convert_ids_to_tokens(input_id)
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output = list(input_ids)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(input_ids) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes: 
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token_id = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token_id = tokenizer.mask_token_id
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token_id = input_ids[index]
            # 10% of the time, replace with random word
            else:
                masked_token_id = rng.randint(0, tokenizer.vocab_size - 1)

        output[index] = masked_token_id

        MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "id"])

        masked_lms.append(MaskedLmInstance(index=index, id=input_ids[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index) ## size=[num_to_predict]

    masked_lm_positions = []
    masked_lm_ids = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_ids.append(p.id)

    return (output, masked_lm_positions, masked_lm_ids)

def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:
    # `x` contains that one sample from lineflow dataset.
    # Example:
    # {
    #    "id": "075e483d21c29a511267ef62bedc0461",
    #    "answer_key": "A",
    #    "options": {"A": "ignore",
    #      "B": "enforce",
    #      "C": "authoritarian",
    #      "D": "yell at",
    #      "E": "avoid"},
    #    "stem": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?"}
    # }
    
    # Use BertTokenizer to encode (tokenize / indexize) two sentences.
    inputs = tokenizer.encode_plus(
            x["string1"],
            x["string2"],
            add_special_tokens=True,
            max_length=MAX_LEN,
            )
    
    # Output of `tokenizer.encode_plus` is a dictionary.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # For BERT, we need `attention_mask` along with `input_ids` as input.
    attention_mask = [1] * len(input_ids)
    # We are going to pad sequences.
    padding_length = MAX_LEN - len(input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    input_ids, masked_lm_positions, masked_lm_ids = create_masked_lm_predictions(input_ids, masked_lm_prob,
                                 max_predictions_per_seq, tokenizer, rng)
    
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    padding_length = max_predictions_per_seq - len(masked_lm_positions)
    masked_lm_positions = masked_lm_positions + ([0] * padding_length)
    masked_lm_ids = masked_lm_ids + ([pad_id] * padding_length)
    masked_lm_weights = masked_lm_weights + ([0.0] * padding_length)

    assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)
    assert len(masked_lm_positions) == max_predictions_per_seq, "Error with input length {} vs {}".format(len(masked_lm_positions), max_predictions_per_seq)
    assert len(masked_lm_ids) == max_predictions_per_seq, "Error with input length {} vs {}".format(len(masked_lm_ids), max_predictions_per_seq)
    
    # Just a python list to `torch.tensor`
    label = torch.tensor(int(x["quality"])).long()
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    masked_lm_positions = torch.tensor(masked_lm_positions)
    masked_lm_ids = torch.tensor(masked_lm_ids)
    masked_lm_weights = torch.tensor(masked_lm_weights)

    # What we return will one instance in batch which `LightningModule.train_step` receives.
    return {
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "masked_lm_weights": masked_lm_weights,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids
            }
  
  
def nonefilter(dataset):
    filtered = []
    for x in dataset:
        if x["string1"] is None:
            continue
        if x["string2"] is None:
            continue
        filtered.append(x)
    return lf.Dataset(filtered)


def get_dataloader():
    # Load datasets (this runs download script for the first run)
    train = lfds.MsrParaphrase("train")
    test = lfds.MsrParaphrase("test")

    # There are some empty entities. Just remove them quickly.
    train = nonefilter(train)
    test = nonefilter(test)

    # Just split train dataset into train and val, so that we can use val for early stopping.
    train, val = lfcv.split_dataset_random(train, int(len(train) * 0.8), seed=42)
    batch_size = 3

    # Now the BERT Tokenizer comes!
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # The actual preprocessing is in this `preprocess` function. (it is defined above.)
    preprocessor = partial(preprocess, tokenizer)

    # Apply the preprocessing and make pytorch dataloaders.
    train_dataloader = DataLoader(
            train.map(preprocessor),
            sampler=RandomSampler(train),
            batch_size=batch_size
            )
    val_dataloader = DataLoader(
            val.map(preprocessor),
            sampler=SequentialSampler(val),
            batch_size=batch_size
            )
    test_dataloader = DataLoader(
            test.map(preprocessor),
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader