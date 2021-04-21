from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch

# # Define FPN
# fpn = resnet_fpn_backbone(
#     backbone_name='resnet18', pretrained=True, trainable_layers=5)



# x = torch.Tensor(1,3,512,512)
# outputs = fpn(x)

# # print(type(outputs))

# for i in outputs.items():
#     print(i[1].shape)

#     # >>> # returns
#     #     >>>   [('0', torch.Size([1, 256, 16, 16])),
#     #     >>>    ('1', torch.Size([1, 256, 8, 8])),
#     #     >>>    ('2', torch.Size([1, 256, 4, 4])),
#     #     >>>    ('3', torch.Size([1, 256, 2, 2])),
#     #     >>>    ('pool', torch.Size([1, 256, 1, 1]))]

import numpy as np
gt_sbj = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])


gt_obj = torch.tensor([[11,12,13],
                  [14,15,16],
                  [17,18,19]])




sbj_boxes = torch.tensor([[1,2,3],[7,8,9]])
obj_boxes = torch.tensor([[11,12,13],[14,15,16]])


a = torch.tensor([torch.where(torch.all(gt_sbj == x, dim=1))[0].item() for x in sbj_boxes])  
b = torch.tensor([torch.where(torch.all(gt_obj == x, dim=1))[0].item() for x in obj_boxes]) 
print(a)
print(b)

mask = a==b

print(~mask)
print(mask.to(dtype=torch.int64))

# sbj = torch.tensor([[1,2,3],
#                   [1,2,3],
#                   [7,8,9]])

# print(torch.unique(sbj, dim=0))

