import numpy as np
import torch

def mobilenet(batch_size):
    from torchvision.models import mobilenet_v2 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def swin_transformer(batch_size):
    from timm.models.swin_transformer import SwinTransformer
    model = SwinTransformer()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def bert(batch_size):
    from .bert_config import BertConfig
    from .pytorch_bert import BertModel

    # from transformers import BertConfig, BertModel
    config = BertConfig(vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=128,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                batch_size=batch_size)
    model = BertModel(config)
    input_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    token_type_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    attention_mask = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    masked_lm_labels = None # torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    next_sentence_label = None # torch.LongTensor(np.ones([config.batch_size]))
    inputs = (input_ids, attention_mask, token_type_ids)
    # inputs = (input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
    return model, inputs

def bert_v0(batch_size):
    from transformers import BertConfig, BertModel
    config = BertConfig(vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=128,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                batch_size=batch_size)
    model = BertModel(config)
    input_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    token_type_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    attention_mask = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    masked_lm_labels = None # torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    next_sentence_label = None # torch.LongTensor(np.ones([config.batch_size]))
    inputs = (input_ids, attention_mask, token_type_ids)
    # inputs = (input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
    return model, inputs

def vit(batch_size):
    from timm.models import vit_small_patch32_224 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def BSRN(batch_size):
    from .BSRN import BSRN
    model = BSRN()
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def NAFNet(batch_size):
    from .nafnet import NAFNet
    model = NAFNet(3, 16, 1, [1, 1, 1], [1, 1, 1])
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def Restormer(batch_size):
    from .restormer import Restormer
    model = Restormer()
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def mobilevit(batch_size):
    from .mobilevit import mobilevit_s
    model = mobilevit_s()
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def NeRF(batch_size):
    from .mlp import MLP
    model = MLP(batch_size=1920*1080, in_dim=64, out_dim=3, hidden_dim=64, n_layers=7)
    input = torch.randn(1920*1080, 64)
    return model, (input, )

def Conformer(batch_size):
    # from torchaudio.models import Conformer # use the next line if torchaudio doesn't works
    from .Conformer import Conformer
    num_frame = 512
    input_dim = 512
    model = Conformer(input_dim=input_dim, num_heads=8, ffn_dim=512, num_layers=12, depthwise_conv_kernel_size=31)
    lengths = torch.LongTensor([num_frame for _ in range(batch_size)])
    input = torch.randn(batch_size, num_frame, input_dim)
    return model, (input, lengths)
