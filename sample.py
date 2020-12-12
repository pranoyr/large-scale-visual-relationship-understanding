from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

# Define FPN
fpn = resnet_fpn_backbone(
    backbone_name='resnet18', pretrained=True, trainable_layers=5)



x = torch.Tensor(1,3,512,512)
outputs = fpn(x)

# print(type(outputs))

for i in outputs.items():
    print(i[1].shape)

    # >>> # returns
    #     >>>   [('0', torch.Size([1, 256, 16, 16])),
    #     >>>    ('1', torch.Size([1, 256, 8, 8])),
    #     >>>    ('2', torch.Size([1, 256, 4, 4])),
    #     >>>    ('3', torch.Size([1, 256, 2, 2])),
    #     >>>    ('pool', torch.Size([1, 256, 1, 1]))]

