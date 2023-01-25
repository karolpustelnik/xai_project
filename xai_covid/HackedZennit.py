import torch 

from zennit.canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from zennit.torchvision import ResNetBottleneckCanonizer, ResNetBasicBlockCanonizer
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from zennit.layer import Sum

class HackBottleneckCanonizer(ResNetBottleneckCanonizer):
    def __init__(self, overwrite_names):
        AttributeCanonizer.__init__(self, self.get_attribute_map(overwrite_names))
        
    
    @classmethod
    def get_attribute_map(cls, overwrite_names):
        
        def _attribute_map(name, module):
            if isinstance(module, ResNetBottleneck):
                if name in overwrite_names:
                    attributes = {
                        'forward': cls.forward_no_grad.__get__(module),
                        'canonizer_sum': Sum(),
                    }
#                    print(name)
                    return attributes
                else:
                    attributes = {
                        'forward': cls.forward.__get__(module),
                        'canonizer_sum': Sum(),
                    }
#                     print("not", name)
                    return attributes
                return None
        return _attribute_map
    
    @staticmethod
    def forward_no_grad(self, x):
        '''
        Modified Bottleneck forward for HackResNet.
        This forward doesn't propagate gradient through skip connections of given layers.
        '''
        identity = x.clone().detach()
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class HackCanonizer(CompositeCanonizer):
    def __init__(self, grad_omit_skips):
        super().__init__((
            SequentialMergeBatchNorm(),
            HackBottleneckCanonizer(grad_omit_skips),
            ResNetBasicBlockCanonizer(),
        ))


def get_canonizer(conditions):
    masked_skips = set()
    for condition in conditions:
        for layer_name in condition.keys():
            if layer_name.startswith("backbone.layer"):
                masked_skips.add(layer_name[:len("bacbkone.layer") + 3])
            if layer_name.startswith("model.layer"):
                masked_skips.add(layer_name[:len("model.layer") + 3])
    print(masked_skips)
    return [HackCanonizer(list(masked_skips))]
