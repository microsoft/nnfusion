import numpy as np
import torch


def resnet(batch_size):
    from torchvision.models import resnet18 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def mobilenet(batch_size):
    from torchvision.models import mobilenet_v2 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def shufflenet(batch_size):
    from torchvision.models import shufflenet_v2_x1_0 as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def squeezenet(batch_size):
    from .squeezenet import SqueezeNet as Net
    model = Net()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def swin_transformer(batch_size):
    # from .swin_transformer import SwinTransformer
    from timm.models.swin_transformer import SwinTransformer
    model = SwinTransformer()
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def EDSR(batch_size):
    from .EDSR import EDSR
    model = EDSR(num_channels=3, base_channel=64, num_residuals=4, upscale_factor=4)
    input = torch.randn(batch_size, 3, 512, 512)
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

def bert_large(batch_size):
    from .bert_config import BertConfig
    from .pytorch_bert import BertModel

    # from transformers import BertConfig, BertModel
    config = BertConfig(vocab_size=30522,
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                max_position_embeddings=512,
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

def transformer(batch_size):
    from transformers import (BertConfig, EncoderDecoderConfig,
                              EncoderDecoderModel)
    config = BertConfig(vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=128,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                batch_size=batch_size)
    config2 = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
    model = EncoderDecoderModel(config2)
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

def localvit(batch_size):
    from .LocalViT import LocalViT as Net
    model = Net(image_size=224, patch_size=16, num_classes=1000)
    input = torch.randn(batch_size, 3, 224, 224)
    return model, (input, )

def crnn(batch_size):
    from .crnn import CRNN
    model = CRNN()
    input = torch.randn(batch_size, 3, 32, 200)
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
