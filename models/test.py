import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

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
    
test_tensor = torch.rand(8, 1, 512, 512)
efnet = EffNet(use_pretrained=False)
efnet_output = efnet(test_tensor)
print(efnet(test_tensor).shape)

encoder_layer = nn.TransformerEncoderLayer(d_model=2304, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = transformer_encoder(efnet_output)


print(out.shape)