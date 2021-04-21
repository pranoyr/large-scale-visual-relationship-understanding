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
                  [1,2,3],
                  [14,2,3]])


gt_obj = torch.tensor([[11,12,13],
                  [14,15,16],
                  [14,15,16]])




sbj_boxes = torch.tensor([[1,2,3], [7,8,9]])
obj_boxes = torch.tensor([[14,15,16], [11,12,13]])


# a = torch.tensor([torch.where(torch.all(gt_sbj == x, dim=1)) for x in sbj_boxes])  
a = torch.where(torch.all(gt_sbj == sbj_boxes[0], dim=1))
b = torch.where(torch.all(gt_obj == obj_boxes[0], dim=1))

# b = torch.tensor([torch.where(torch.all(gt_obj == x, dim=1))[0].item() for x in obj_boxes]) 
print(a)
print(b)

a = np.intersect1d(a[0],b[0])
print(a)

if a.any():
    print(a[0])

# print(b)

# mask = a==b

# print(~mask)
# print(mask.to(dtype=torch.int64))

# sbj = torch.tensor([[1,2,3],
#                   [1,2,3],
#                   [7,8,9]])

# print(torch.unique(sbj, dim=0))

