import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
from torchmetrics.functional import precision_recall, accuracy
from torchmetrics.functional import auc
from torchmetrics.functional import f1_score
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)



class Fetal(data.Dataset):
    def __init__(self, videos_path, root, ann_path, transform=None, target_transform=None):
        
        
        self.videos = pd.read_csv(videos_path)
        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, frame_positions, labels)
        """
        images = list() # list of images
        labels = list() # list of labels
        vid = self.videos.iloc[index][0]
        print(vid)
        vid_len = self.database.query('video == @vid')['frames_n'].iloc[0]
        print(vid_len)
        #print(self.database.query(vid[0]))
        #index, #class, #video, #frames_n
        # images
        for frame in range(vid_len):
            image = self._load_image(self.data_path  + vid + f'_{frame+1}' + '.png')
            # transform image              
            if self.transform is not None:
                image = self.transform(image)
            images.append(image) # append image to list
            #print(f'{vids[0]}_' + f'{frame}')
            idx = f'{vid}_' + f'{frame+1}'
            label = self.database.query('index ==@idx')['Class'].iloc[0]
            #print(label)
            label = torch.tensor(label) # convert to tensor
            labels.append(label) # append label to list

        # target
        frames_position = [i+1 for i in range(vid_len)]
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        images = torch.stack(images)
        labels = torch.stack(labels)
        images = images.permute(1, 0, 2, 3)
        frames_position = torch.tensor(frames_position)
        return images, frames_position, labels

    def __len__(self):
        return len(self.videos)
    

transform = transforms.Compose([transforms.Resize((450, 600)),
                transforms.Grayscale(num_output_channels=1),
                transforms.Pad((0, 0, 0, 150), fill = 0, padding_mode = 'constant'),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.1354949, std=0.18222201)])
dataset = Fetal(videos_path = '/data/kpusteln/fetal/videos_val.csv',
                root = '/data/kpusteln/fetal/fetal_extracted/', 
                ann_path = "/data/kpusteln/fetal/fetal_val_final.csv", 
                transform = transform)


data_loader_test = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1,
    drop_last=True,
    shuffle = False,
)

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.resnet import resnet18
from torchvision.models import resnet50, resnet101
from efficientnet_pytorch import EfficientNet
from transformers import LongformerModel, LongformerConfig
import torch.nn.functional as F



class EffNet(torch.nn.Module):
    def __init__(self, out_features = 7, extract = True, freeze = True, unfreeze_last_layers = True):
        super(EffNet, self).__init__()
        self.out_features = out_features
        self.extract = extract
        
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6', in_channels = 1, num_classes=self.out_features)
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
                 max_position_embeddings=2 * 60 * 60,
                 num_attention_heads=12,
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

    def __init__(self):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(EFL, self).__init__()
        self._construct_network()

    def _construct_network(self, task_type = 'cls'):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.backbone = EffNet(freeze = True, unfreeze_last_layers = True).to('cuda:0', dtype=torch.float32)
        self.num_classes = self.backbone.out_features
        embed_dim = 2304 ## featury z EfficientNet
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)).to('cuda:1', dtype=torch.float32)
        self.hidden_size = 2304 
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

        x, position_ids, _ = x
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

        #x = self.backbone(x)
        #print(f'shape after backbone: {x.shape}')
         # temporal encoder (Longformer)
        x = x.reshape(B, F, -1)
        x = x.to('cuda:1')
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device='cuda:1')
        #print(f' attenstion mask shape: {attention_mask.shape}')
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #print(f'cls token shape" {cls_tokens.shape}')
        #print(x.get_device())
        x = torch.cat((cls_tokens, x), dim=1)
        #print(f'concatenated x shape: {x.shape}')
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
            x = x[:, 0]
            x = self.mlp_head(x)
        return x
    
    
#model = EFL()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_tensor = torch.randn(1, 1, 600, 512, 512) # B, C, F, H, W

ids = torch.arange(0, test_tensor.size(2)) # fake ids
ids = ids.unsqueeze(0)
x = (test_tensor, ids, ids)
# model = EffNet()
# print(len(model.backbone._blocks[43:]))
model = EFL()
#print(model(x).shape)
print(torch.cuda.device_count())
criterion_cls = torch.nn.CrossEntropyLoss()
for j, (images, frames, labels) in enumerate(data_loader_test):
    output = model((images, frames, frames))
    labels = labels.cuda('cuda:1')
    loss = criterion_cls(output, labels)
    print(f'labels: {labels.shape}')
    print(f'output: {output.shape}')
    precision, recall = precision_recall(output, labels, average = 'macro', num_classes = 7)
    f1 = f1_score(output, labels, average = 'macro', num_classes = 7)
    print(loss)
    break