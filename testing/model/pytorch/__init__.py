import torch
import numpy as np

def resnet(batch_size):
    from torchvision.models import resnet18 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def swin_transormer(batch_size):
    from .swin_transformer import SwinTransformer
    model = SwinTransformer()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def EDSR(batch_size):
    from .EDSR import EDSR
    model = EDSR(num_channels=3, base_channel=64, num_residuals=4, upscale_factor=4)
    input = torch.randn(batch_size, 3, 512, 512)
    return model, (input, )

def bert(batch_size):
    from .pytorch_bert import BertModel
    from .bert_config import BertConfig
    # from transformers import BertModel, BertConfig
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
    from vit_pytorch import ViT
    model = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    input = torch.randn(batch_size, 3, 256, 256)
    return model, (input, )

def crnn(batch_size):
    from .crnn import CRNN
    model = CRNN()
    input = torch.randn(batch_size, 3, 32, 200)
    return model, (input, )
