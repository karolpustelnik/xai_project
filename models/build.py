# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer, ResNet101, EffNet
from .effnet_longformer import EFL
from .UniNet import UniNetB6, UniNetB1
from .vit import ViT
#from .resnet import resnet34, resnet50, resnet101 #not pretraiend
from .resnetv2 import ResNet50, ResNet101
from .resnet_attention import resnet50
def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE_SWIN,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                task_type=config.MODEL.TASK_TYPE,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    #elif model_type == 'resnet_34':
    #    model = resnet34()
    elif model_type == 'resnet_50':
        model = ResNet50(out_features=config.MODEL.NUM_CLASSES, freeze = config.MODEL.FREEZE, in_channels = config.MODEL.IN_CHANNELS)
    elif model_type == 'resnet_101':
        model = ResNet101(out_features=config.MODEL.NUM_CLASSES, freeze = config.MODEL.FREEZE, in_channels = config.MODEL.IN_CHANNELS)
    elif model_type == 'resnet_50_attention':
        model = resnet50()
        
    elif model_type == 'effnet':
        model = EffNet(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'efl':
        model = EFL(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'uninet':
        model = UniNetB1(out_features = config.MODEL.NUM_CLASSES)
    elif model_type == 'vit':
         model = ViT(512, 32, 1, 1024, 6, 16, 2048, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.)
        
    return model
