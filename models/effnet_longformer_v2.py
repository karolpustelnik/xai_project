import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.resnet import resnet18
from torchvision.models import resnet50, resnet101
from efficientnet_pytorch import EfficientNet
from transformers import LongformerModel, LongformerConfig, LongformerForTokenClassification
import torch.nn.functional as F



class EffNet(torch.nn.Module):
    def __init__(self, out_features = 7, use_pretrained = True, extract = True, freeze = True, unfreeze_last_layers = True):
        super(EffNet, self).__init__()
        self.out_features = out_features
        self.extract = extract
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6', in_channels = 1, num_classes=self.out_features)
        if use_pretrained:
            model = torch.load('/data/kpusteln/Fetal-RL/swin-transformer/output/effnet_reg_v2_abdomen/default/ckpt_epoch_61.pth')['model']
            for key in list(model.keys()):
                if 'backbone' in key:
                    model[key.replace('backbone.', '')] = model.pop(key) # remove prefix backbone.
            self.backbone.load_state_dict(model)
        if self.extract:    ## extract features for the transformer, ignore last layer
            self.backbone._fc = torch.nn.Identity()
        if freeze:
            for param in self.backbone.parameters():
                    param.requires_grad = False
                
        if unfreeze_last_layers:
            for param in self.backbone._blocks[44:].parameters():
                    param.requires_grad = True
                
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.backbone(x)
        
        return x
    
    

#model = EffNet()

#test_img = torch.randn(1, 1, 512, 512)
#print(model(test_img).shape)

class LongformerModel(LongformerModel):

    def __init__(self,
                 embed_dim=2304,
                 max_position_embeddings=1024,
                 num_attention_heads=6,
                 num_hidden_layers=3,
                 attention_mode='sliding_chunks',
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):

        self.config = LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1, ] * num_hidden_layers
        self.config.attention_window = [256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(LongformerModel, self).__init__(self.config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None

def pad_to_window_size_local(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                             one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer self-attention.
    Based on _pad_to_window_size from https://github.com/huggingface/transformers:
    https://github.com/huggingface/transformers/blob/71bdc076dd4ba2f3264283d4bc8617755206dccd/src/transformers/models/longformer/modeling_longformer.py#L1516
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids.permute(0, 2, 1), (0, padding_len), value=pad_token_id).permute(0, 2, 1)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    position_ids = F.pad(position_ids, (1, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask, position_ids

class EFL(nn.Module):
    """
    Efficient net backbone + longformer --> classification or regression
    """

    def __init__(self, out_features = 7, task_type = 'reg'):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.out_features = out_features
        self.task_type = task_type
        super(EFL, self).__init__()
        self._construct_network(self.out_features, self.task_type)

    def _construct_network(self, out_features = 7, task_type = 'reg'):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.backbone = EffNet(out_features = out_features, freeze = True, unfreeze_last_layers = True).to('cuda:0', dtype=torch.float32)
        self.num_classes = self.backbone.out_features
        self.embed_dim = 2304 ## featury z EfficientNet
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)).to('cuda:1', dtype=torch.float32)
        self.hidden_size = 2304
        #self.temporal_encoder = LongformerModel(embed_dim = self.embed_dim).to('cuda:1', dtype=torch.float32)
        self.temporal_encoder = LongformerModel().to('cuda:1', dtype=torch.float32)
        self.task_type = task_type
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.temporal_encoder.config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, 1 if self.task_type == 'reg' else self.num_classes)
        ).to('cuda:1')

    def forward(self, x):

        x, position_ids = x #x: B - batch size, C - channels, F - frames, H - height, W - widt
        x = x.to('cuda:0', dtype=torch.float32)
        position_ids = position_ids.to('cuda:1', dtype=torch.float32)
        # spatial backbone
        B, C, F, H, W = x.shape # B - batch size, C - channels, F - frames, H - height, W - width
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        #print(x.dtype)
        chunk_size = 8
        #n_chunks = int(B*F/chunk_size) + 1
        #print(n_chunks)
        x = torch.split(x, chunk_size, dim=0)
        backbone_output = []
        for i, chunk in enumerate(x):
            #print(i)
            chunk = self.backbone(chunk)
            #print(torch.cuda.memory_allocated(0))
            #print(chunk.shape)
            chunk = chunk.to('cuda:1', dtype=torch.float32)
            #print(torch.cuda.memory_allocated(0))
            chunk = chunk.unsqueeze(0)
            backbone_output.append(chunk)
            
        x = torch.cat(backbone_output, dim=1)

         # temporal encoder (Longformer)
        x = x.reshape(B, F, -1)
        x = x.to('cuda:1')
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device='cuda:1')
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to('cuda:1', dtype=torch.float32)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device='cuda:1')
        token_type_ids[:, 0] = 1
        #print(x.shape)
        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1
        #print(position_ids)
        #print(f'position_ids shape: {position_ids.shape}')
        x.to('cuda:1', dtype=torch.float32)
        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        # MLP head
        
        x = x["last_hidden_state"]
        #print(x.shape)
        if self.task_type == 'cls':
            x = x[:, 1:F+1, :]  # extract frames
            x = self.mlp_head(x)
            x = x.permute(0, 2, 1)
        elif self.task_type == 'reg':
            x = self.mlp_head(x[:, 0])
        return x
    
# B - batch size, C - channels, F - frames, H - height, W - width 


#test_tensor = torch.rand(1, 1, 240, 512, 512)

#positions = torch.tensor([[i for i in range(240)]])

#model = EFL(out_features = 1)

#print(model((test_tensor, positions, torch.tensor([17]))).shape)
