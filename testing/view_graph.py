import torch
import numpy as np
from torchviz import make_dot

def profile_resnet():
    from torchvision.models import resnet18 as Net
    model = Net().cuda(1)
    input = torch.randn(64, 3, 224, 224).cuda(1)
    vis_graph = make_dot(model(input), params=dict(model.named_parameters()))
    vis_graph.view()

def profile_CRNN():
    from model.pytorch.crnn import CRNN
    model = CRNN()
    input = torch.randn(64, 3, 32, 32)
    vis_graph = make_dot(model(input), params=dict(model.named_parameters()))
    vis_graph.view()

def profile_swin_transormer():
    from model.pytorch.swin_transformer import SwinTransformer
    model = SwinTransformer()
    input = torch.randn(64, 3, 224, 224)
    vis_graph = make_dot(model(input), params=dict(model.named_parameters()))
    vis_graph.view()

def profile_EDSR():
    from model.pytorch.EDSR import EDSR
    model = EDSR(num_channels=3, base_channel=64, num_residuals=4, upscale_factor=4)
    input = torch.randn(1, 3, 512, 512)
    vis_graph = make_dot(model(input), params=dict(model.named_parameters()))
    vis_graph.view()

def profile_bert():
    from model.pytorch.pytorch_bert import BertForPreTraining
    from model.pytorch.bert_config import BertConfig
    config = BertConfig(vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=128,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                batch_size=64)
    model = BertForPreTraining(config)
    input_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    token_type_ids = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    attention_mask = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    masked_lm_labels = torch.LongTensor(np.ones([config.batch_size, config.max_position_embeddings]))
    next_sentence_label = torch.LongTensor(np.ones([config.batch_size]))
    inputs = (input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
    vis_graph = make_dot(model(*inputs), params=dict(model.named_parameters()))
    vis_graph.view()

def profile_vit():
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
    input = torch.randn(64, 3, 256, 256)
    vis_graph = make_dot(model(input), params=dict(model.named_parameters()))
    vis_graph.view()

if __name__ == "__main__":
    profile_resnet()
