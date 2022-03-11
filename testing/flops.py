import torch
import numpy as np
import thop

def get_base_op_list(model):
    result = []
    for name, m in model.named_children():
        if not any(m.named_children()):
            result.append(m)
        else:
            result.extend(get_base_op_list(m))
    return result

class MyHook:
    def __init__(self) -> None:
        self.read = 0
        self.write = 0

    def func(self, module, fea_in, fea_out):
        num_read = 0
        num_write = 0
        print(module)
        for tensor in fea_in:
            num_read += np.prod(tensor.shape)
        for tensor in fea_out:
            num_write += np.prod(tensor.shape)
        self.read += num_read
        self.write += num_write
        # print(module, num_read, num_write)
        return None

def profile_resnet():
    from torchvision.models import vgg16 as Net
    model = Net()
    input = torch.randn(64, 3, 224, 224)
    macs, params = thop.profile(model, inputs=(input, ))
    op_list = get_base_op_list(model)
    hook = MyHook()
    for op in op_list:
        op.register_forward_hook(hook.func)
    model(input)
    print("OpCount", len(op_list))
    print("FLOPs", 2 * macs, "Params", params)
    print("Read", hook.read, "Write", hook.write)

def profile_swin_transormer():
    from model.pytorch.swin_transformer import SwinTransformer
    model = SwinTransformer()
    input = torch.randn(64, 3, 224, 224)
    macs, params = thop.profile(model, inputs=(input, ))
    op_list = get_base_op_list(model)
    hook = MyHook()
    for op in op_list:
        op.register_forward_hook(hook.func)
    model(input)
    print("OpCount", len(op_list))
    print("FLOPs", 2 * macs, "Params", params)
    print("Read", hook.read, "Write", hook.write)

def profile_EDSR():
    from model.pytorch.EDSR import EDSR
    model = EDSR(num_channels=3, base_channel=64, num_residuals=4, upscale_factor=4)
    input = torch.randn(1, 3, 512, 512)
    macs, params = thop.profile(model, inputs=(input, ))
    op_list = get_base_op_list(model)
    hook = MyHook()
    for op in op_list:
        op.register_forward_hook(hook.func)
    model(input)
    print("OpCount", len(op_list))
    print("FLOPs", 2 * macs, "Params", params)
    print("Read", hook.read, "Write", hook.write)

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
    macs, params = thop.profile(model, inputs=inputs)
    op_list = get_base_op_list(model)
    hook = MyHook()
    for op in op_list:
        op.register_forward_hook(hook.func)
    model(*inputs)
    print("OpCount", len(op_list))
    print("FLOPs", 2 * macs, "Params", params)
    print("Read", hook.read, "Write", hook.write)

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
    macs, params = thop.profile(model, inputs=(input, ))
    op_list = get_base_op_list(model)
    hook = MyHook()
    for op in op_list:
        op.register_forward_hook(hook.func)
    model(input)
    print("OpCount", len(op_list))
    print("FLOPs", 2 * macs, "Params", params)
    print("Read", hook.read, "Write", hook.write)

if __name__ == "__main__":
    profile_swin_transormer()
